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
        logging.info('running %d walkers for %d steps', self.k, nsteps)

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
    def __init__(self, path=workdir / 'mcmc' / 'chain.hdf',
                 expdata_path="./exp_data.dat",
                 model_parafile="./model.dat",
                 training_data_path="./training_data"
    ):
        self.path = path
        self.path.parent.mkdir(exist_ok=True)

        # load the model parameter file
        self.pardict = parse_model_parameter_file(model_parafile)
        self.ndim = len(self.pardict.keys())
        self.min = []
        self.max = []
        for par, val in self.pardict.items():
            self.min.append(val[1])
            self.max.append(val[2])
        self.min = np.array(self.min)
        self.max = np.array(self.max)

        # load the experimental data to be fit
        self.expdata, self.expdata_cov = self._read_in_exp_data(expdata_path)
        self.nobs = self.expdata.shape[0]

        # setup the emulator
        self.emu = Emulator(
            training_set_path=training_data_path,
            parameter_file=model_parafile
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
            #model_Y, model_cov = self._toy_model(X[inside])

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

    #def _toy_model(self, X):
    #    nsamples = X.shape[0]
    #    xx = np.linspace(-5, 5, 21)
    #    y1 = np.zeros([nsamples, 21])
    #    y2 = np.zeros([nsamples, 21])
    #    for j in range(nsamples):
    #        y1[j, :] = X[j, 0]*np.exp(-X[j, 1]*xx**2.)
    #        y2[j, :] = X[j, 2]*np.cosh(X[j, 3]*xx)
    #    Y = np.concatenate((y1, y2), axis=1)
    #    Y_cov = np.zeros([nsamples, 42, 42])
    #    for j in range(nsamples):
    #        for i in range(42):
    #            Y_cov[j, i, i] = Y[j, i]*0.0001
    #    return Y, Y_cov

    def _read_in_exp_data(self, filepath):
        """This function reads in exp data and compute the covarance matrix"""
        data = np.loadtxt(filepath)
        nobs = data.shape[0]
        data_cov = np.zeros([nobs, nobs])
        for i in range(nobs):
            data_cov[i, i] = data[i, 1]**2.
        return data[:, 0], data_cov


    def random_pos(self, n=1):
        """
        Generate `n` random positions in parameter space.

        """
        return np.random.uniform(self.min, self.max, (n, self.ndim))


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
            storechain=False
        )[0]
        sampler.reset()
        logging.info('burn-in complete, starting production')

        sampler.run_mcmc(X0, nsteps, status=status)

        logging.info('writing chain to file')

        true_val = [0.2, 0.5, 0.7, 0.3]
        fig, axlist = plt.subplots(self.ndim, 1, sharex=True)
        for idim in range(self.ndim):
            for iwalker in range(nwalkers):
                axlist[idim].plot([0, nsteps], [true_val[idim], true_val[idim]], '-r')
                axlist[idim].plot(sampler.chain[iwalker, :, idim], '-k', alpha=0.1)
        axlist[0].set_xlim([0, nsteps])
        plt.show()

        samples = sampler.chain[:, :, :].reshape((-1, self.ndim))

        import corner
        fig = corner.corner(samples, labels=["$A$", "$B$", "$C$", "D"])
        plt.savefig("test.png")
        results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                      zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        print(list(results))

        fig = plt.figure()
        plt.errorbar(range(len(self.expdata)), self.expdata,
                     np.sqrt(self.expdata_cov.diagonal()),
                     linestyle='', marker='o', color='k')
        xl = np.linspace(-5, 5, 21)
        for a, b, c, d in samples[np.random.randint(len(samples), size=100)]:
            y1 = a*np.exp(-b*xl**2.)
            y2 = c*np.cosh(d*xl)
            Y = np.concatenate((y1, y2))
            plt.plot(range(len(Y)), Y, linestyle='-', color='g', alpha=0.1)
        plt.show()



def main():
    parser = argparse.ArgumentParser(
            description='Markov chain Monte Carlo',
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
