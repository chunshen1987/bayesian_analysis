"""
Markov chain Monte Carlo model calibration using the `affine-invariant ensemble
sampler (emcee) <http://dfm.io/emcee>`_.

This module must be run explicitly to create the posterior distribution.
Run ``python -m src.mcmc --help`` for complete usage information.

On first run, the number of walkers and burn-in steps must be specified, e.g.
::

    python -m src.mcmc --nwalkers 500 --nburnsteps 100 200

would run 500 walkers for 100 burn-in steps followed by 200 production steps.
This will create the HDF5 file :file:`mcmc/chain.hdf` (default path).

On subsequent runs, the chain resumes from the last point and the number of
walkers is inferred from the chain, so only the number of production steps is
required, e.g. ::

    python -m src.mcmc 300

would run an additional 300 production steps (total of 500).

To restart the chain, delete (or rename) the chain HDF5 file.
"""

import argparse
import logging

import emcee
import h5py
import numpy as np
from scipy.linalg import lapack
import matplotlib.pyplot as plt

from . import workdir, parse_model_parameter_file
from .emulator import Emulator


def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.  The
    normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )

    return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()


class LoggingEnsembleSampler(emcee.EnsembleSampler):
    def run_mcmc(self, X0, nsteps, status=None, **kwargs):
        """
        Run MCMC with logging every 'status' steps (default: approx 10% of
        nsteps).

        """
        logging.info('running %d walkers for %d steps', self.nwalkers, nsteps)

        if status is None:
            status = nsteps // 10

        for n, result in enumerate(
                self.sample(X0, iterations=nsteps, **kwargs),
                start=1
        ):
            if n % status == 0 or n == nsteps:
                af = self.acceptance_fraction
                logging.info(
                    'step %d: acceptance fraction: '
                    'mean %.4f, std %.4f, min %.4f, max %.4f',
                    n, af.mean(), af.std(), af.min(), af.max()
                )

        return result


class Chain:
    """
    High-level interface for running MCMC calibration and accessing results.

    Currently all design parameters except for the normalizations are required
    to be the same at all beam energies.  It is assumed (NOT checked) that all
    system designs have the same parameters and ranges (except for the norms).

    """
    def __init__(self, mcmc_path=workdir / 'mcmc' / 'chain.h5',
                 expdata_path="./exp_data.dat",
                 model_parafile="./model.dat",
                 training_data_path="./training_data",
                 npc=10,
    ):
        logging.info('Initializing MCMC ...')
        self.mcmc_path = mcmc_path
        self.mcmc_path.parent.mkdir(exist_ok=True)
        logging.info('Final Markov Chain results will be saved in {}'.format(
            self.mcmc_path)
        )

        # load the model parameter file
        logging.info('Loading the model parameters space from {} ...'.format(
            model_parafile)
        )
        self.pardict = parse_model_parameter_file(model_parafile)
        self.ndim = len(self.pardict.keys())
        self.label = []
        self.min = []
        self.max = []
        for par, val in self.pardict.items():
            self.label.append(val[0])
            self.min.append(val[1])
            self.max.append(val[2])
        self.min = np.array(self.min)
        self.max = np.array(self.max)

        # load the experimental data to be fit
        logging.info(
            'Loading the experiment data from {} ...'.format(expdata_path))
        self.expdata, self.expdata_cov = self._read_in_exp_data(expdata_path)
        self.nobs = self.expdata.shape[0]
        self.closureTestFalg = False

        # setup the emulator
        logging.info('Initializing emulators for the training model ...')
        self.emu = Emulator(
            training_set_path=training_data_path,
            parameter_file=model_parafile,
            npc=npc
        )


    def log_posterior(self, X, extra_std_prior_scale=.05):
        """
        Evaluate the posterior at `X`.

        `extra_std_prior_scale` is the scale parameter for the prior
        distribution on the model sys error parameter:

            prior ~ sigma^2 * exp(-sigma/scale)

        """
        X = np.array(X, copy=False, ndmin=2)

        lp = np.zeros(X.shape[0])

        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        nsamples = np.count_nonzero(inside)

        if nsamples > 0:
            # not sure why to use the last parameter for extra std
            extra_std = 0.0*X[inside, -1]

            model_Y, model_cov = self.emu.predict(
                X[inside], return_cov=True, extra_std=extra_std
            )

            # allocate difference (model - expt) and covariance arrays
            dY = np.empty([nsamples, self.nobs])
            cov = np.empty([nsamples, self.nobs, self.nobs])
            dY = model_Y - self.expdata
            # add expt cov to model cov
            cov = model_cov + self.expdata_cov

            # compute log likelihood at each point
            lp[inside] += list(map(mvn_loglike, dY, cov))

            # add prior for extra_std (model sys error)
            lp[inside] += (2*np.log(extra_std + 1e-16)
                           - extra_std/extra_std_prior_scale)

        return lp


    def _read_in_exp_data(self, filepath):
        """This function reads in exp data and compute the covarance matrix"""
        data = np.loadtxt(filepath)
        nobs = data.shape[0]
        data_cov = np.zeros([nobs, nobs])
        for i in range(nobs):
            data_cov[i, i] = (data[i, 2]/data[i, 1])**2.
        return np.log(data[:, 1]), data_cov


    def random_pos(self, n=1):
        """
        Generate `n` random positions in parameter space.

        """
        return np.random.uniform(self.min, self.max, (n, self.ndim))


    def set_closure_test_truth(self, filename):
        self.closureTestFalg = True
        self.trueParams = []
        with open(filename, "r") as parfile:
            for line in parfile:
                line = line.split()
                self.trueParams.append(float(line[1]))
        self.trueParams = np.array(self.trueParams)


    @staticmethod
    def map(f, args):
        """
        Dummy function so that this object can be used as a 'pool' for
        :meth:`emcee.EnsembleSampler`.

        """
        return f(args)


    def run_mcmc(self, nsteps=500, nburnsteps=None, nwalkers=None,
                 status=None):
        """
        Run MCMC model calibration.  If the chain already exists, continue from
        the last point, otherwise burn-in and start the chain.

        """

        logging.info('Starting MCMC ...')
        sampler = LoggingEnsembleSampler(
            nwalkers, self.ndim, self.log_posterior, pool=self
        )

        logging.info('no existing chain found, starting initial burn-in')

        # Run first half of burn-in starting from random positions.
        nburn0 = nburnsteps // 2
        sampler.run_mcmc(
            self.random_pos(nwalkers),
            nburn0,
            status=status
        )
        logging.info('resampling walker positions')
        # Reposition walkers to the most likely points in the chain,
        # then run the second half of burn-in.  This significantly
        # accelerates burn-in and helps prevent stuck walkers.
        X0 = sampler.flatchain[
            np.unique(
                sampler.flatlnprobability,
                return_index=True
            )[1][-nwalkers:]
        ]
        sampler.reset()
        X0 = sampler.run_mcmc(
            X0,
            nburnsteps - nburn0,
            status=status,
        )
        sampler.reset()
        logging.info('burn-in complete, starting production')

        sampler.run_mcmc(X0, nsteps, status=status)

        logging.info('writing chain to file')
        return(sampler)


    def make_plots(self, chains):
        nwalkers, nsteps, ndim = chains.shape
        fig, axlist = plt.subplots(ndim, 1, sharex=True)
        for idim in range(self.ndim):
            for iwalker in range(nwalkers):
                axlist[idim].plot(chains[iwalker, :, idim], '-k', alpha=1./nwalkers)
        axlist[0].set_xlim([0, nsteps])
        plt.show()

        samples = chains[:, :, :].reshape((-1, ndim))

        import corner
        fig = corner.corner(samples, labels=self.label)
        if self.closureTestFalg:
            corner.overplot_lines(fig, self.trueParams, color="C1")
            corner.overplot_points(fig, self.trueParams[None], marker="o",
                                   color="C1")
        plt.savefig("posterior.png")
        results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        if self.closureTestFalg:
            for ipar, par in enumerate(list(results)):
                print("%s = %.4f^{+%.4f}_{-%.4f}, truth = %.4f" % (
                    self.label[ipar], par[0], par[1], par[2],
                    self.trueParams[ipar])
                )
        else:
            for ipar, par in enumerate(list(results)):
                print("%s = %.4f^{+%.4f}_{-%.4f}" % (
                                self.label[ipar], par[0], par[1], par[2])
                )

        fig = plt.figure()
        plt.errorbar(range(len(self.expdata)), self.expdata,
                     np.sqrt(self.expdata_cov.diagonal()),
                     linestyle='', marker='o', color='k')
        model_Y = self.emu.predict(
                samples[np.random.randint(len(samples), size=100)])
        for model_res in model_Y:
            plt.plot(range(len(self.expdata)), model_res, color='r', alpha=0.3)
        plt.show()



def main():
    parser = argparse.ArgumentParser(
            description='Markov Chain Monte Carlo',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--nsteps', type=int, default=500,
        help='number of steps'
    )
    parser.add_argument(
        '--nwalkers', type=int, default=100,
        help='number of walkers'
    )
    parser.add_argument(
        '--nburnsteps', type=int, default=200,
        help='number of burn-in steps'
    )
    parser.add_argument(
        '--status', type=int,
        help='number of steps between logging status'
    )
    parser.add_argument(
        '--exp', type=str, default='./exp_data.dat',
        help="experimental data"
    )
    parser.add_argument(
        '--model_design', type=str,
        default='model_parameter_dict_examples/ABCD.txt',
        help="model parameter filename"
    )
    parser.add_argument(
        '--training_set', type=str,
        default='./training_dataset',
        help="model training set parameters"
    )
    args = parser.parse_args()

    mymcmc = Chain(expdata_path=args.exp, model_parafile=args.model_design,
                   training_data_path=args.training_set)
    mymcmc.run_mcmc(nsteps=args.nsteps, nburnsteps=args.nburnsteps,
                    nwalkers=args.nwalkers, status=args.status)


if __name__ == '__main__':
    main()
