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

from . import workdir


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
    def __init__(self, path=workdir / 'mcmc' / 'chain.hdf'):
        self.path = path
        self.path.parent.mkdir(exist_ok=True)

    def run_mcmc(self, nsteps, nburnsteps=None, nwalkers=None, status=None):
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


def main():
    parser = argparse.ArgumentParser(
            description='Markov chain Monte Carlo',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'nsteps', type=int,
        help='number of steps'
    )
    parser.add_argument(
        '--nwalkers', type=int,
        help='number of walkers'
    )
    parser.add_argument(
        '--nburnsteps', type=int,
        help='number of burn-in steps'
    )
    parser.add_argument(
        '--status', type=int,
        help='number of steps between logging status'
    )
    args = parser.parse_args()

    #Chain().run_mcmc(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
