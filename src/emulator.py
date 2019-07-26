"""
Trains Gaussian process emulators.

When run as a script, allows retraining emulators, specifying the number of
principal components, and other options (however it is not necessary to do this
explicitly --- the emulators will be trained automatically when needed).  Run
``python -m src.emulator --help`` for usage information.

Uses the `scikit-learn <http://scikit-learn.org>`_ implementations of
`principal component analysis (PCA)
<http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
and `Gaussian process regression
<http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_.
"""

import logging

import numpy as np
from os import path
from glob import glob

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.externals import joblib

from . import cachedir, parse_model_parameter_file


class Emulator:
    """
    Multidimensional Gaussian process emulator using principal component
    analysis.

    The model training data are standardized (subtract mean and scale to unit
    variance), then transformed through PCA.  The first `npc` principal
    components (PCs) are emulated by independent Gaussian processes (GPs).  The
    remaining components are neglected, which is equivalent to assuming they
    are standard zero-mean unit-variance GPs.

    This class has become a bit messy but it still does the job.  It would
    probably be better to refactor some of the data transformations /
    preprocessing into modular classes, to be used with an sklearn pipeline.
    The classes would also need to handle transforming uncertainties, which
    could be tricky.

    """
    def __init__(self, training_set_path=".", parameter_file="ABCD.txt",
                 npc=10, nrestarts=0):
        self._load_training_data(training_set_path)

        self.pardict = parse_model_parameter_file(parameter_file)
        self.design_min = []
        self.design_max = []
        for par, val in self.pardict.items():
            self.design_min.append(val[1])
            self.design_max.append(val[2])
        self.design_min = np.array(self.design_min)
        self.design_max = np.array(self.design_max)

        self.npc = npc
        nev, self.nobs = self.model_data.shape

        self.scaler = StandardScaler(copy=False)
        self.pca = PCA(copy=False, whiten=True, svd_solver='full')

        # Standardize observables and transform through PCA.  Use the first
        # `npc` components but save the full PC transformation for later.
        Z = self.pca.fit_transform(
                self.scaler.fit_transform(self.model_data))[:, :npc]

        # Define kernel (covariance function):
        # Gaussian correlation (RBF) plus a noise term.
        ptp = self.design_max - self.design_min
        kernel = (
            1. * kernels.RBF(
                length_scale=ptp,
                length_scale_bounds=np.outer(ptp, (.1, 10))
            ) +
            kernels.WhiteKernel(
                noise_level=.1**2,
                noise_level_bounds=(.01**2, 1)
            )
        )

        # Fit a GP (optimize the kernel hyperparameters) to each PC.
        self.gps = [
            GPR(
                kernel=kernel, alpha=0,
                n_restarts_optimizer=nrestarts,
                copy_X_train=False
            ).fit(self.design_points, z)
            for z in Z.T
        ]

        # Construct the full linear transformation matrix, which is just the PC
        # matrix with the first axis multiplied by the explained standard
        # deviation of each PC and the second axis multiplied by the
        # standardization scale factor of each observable.
        self._trans_matrix = (
            self.pca.components_
            * np.sqrt(self.pca.explained_variance_[:, np.newaxis])
            * self.scaler.scale_
        )

        # Pre-calculate some arrays for inverse transforming the predictive
        # variance (from PC space to physical space).

        # Assuming the PCs are uncorrelated, the transformation is
        #
        #   cov_ij = sum_k A_ki var_k A_kj
        #
        # where A is the trans matrix and var_k is the variance of the kth PC.
        # https://en.wikipedia.org/wiki/Propagation_of_uncertainty

        # Compute the partial transformation for the first `npc` components
        # that are actually emulated.
        A = self._trans_matrix[:npc]
        self._var_trans = np.einsum(
            'ki,kj->kij', A, A, optimize=False).reshape(npc, self.nobs**2)

        # Compute the covariance matrix for the remaining neglected PCs
        # (truncation error).  These components always have variance == 1.
        B = self._trans_matrix[npc:]
        self._cov_trunc = np.dot(B.T, B)

        # Add small term to diagonal for numerical stability.
        self._cov_trunc.flat[::self.nobs + 1] += 1e-4 * self.scaler.var_


    def _inverse_transform(self, Z):
        """
        Inverse transform principal components to observables.
        # Z shape (..., npc)
        # Y shape (..., nobs)

        """
        Y = np.dot(Z, self._trans_matrix[:Z.shape[-1]])
        Y += self.scaler.mean_

        return Y


    def _load_training_data(self, data_path):
        """This function read in training data set at every sample point"""
        self.model_data = []
        self.design_points = []
        for iev in glob(path.join(data_path, "*")):
            with open(path.join(iev, "parameters.txt"), "r") as parfile:
                parameters = []
                for line in parfile:
                    line = line.split()
                    parameters.append(float(line[1]))
            self.design_points.append(parameters)
            temp_data = np.loadtxt(path.join(iev, "output.txt"))
            self.model_data.append(temp_data)
        self.design_points = np.array(self.design_points)
        self.model_data = np.array(self.model_data)


    def predict(self, X, return_cov=False, extra_std=0):
        """
        Predict model output at `X`.

        X must be a 2D array-like with shape ``(nsamples, ndim)``. It is passed
        directly to sklearn :meth:`GaussianProcessRegressor.predict`.

        If `return_cov` is true, return a tuple ``(mean, cov)``, otherwise only
        return the mean.

        The mean is returned as a nested dict of observable arrays, each with
        shape ``(nsamples, n_cent_bins)``.

        The covariance is returned as a proxy object which extracts observable
        sub-blocks using a dict-like interface:

        The shape of the extracted covariance blocks are
        ``(nsamples, n_cent_bins_1, n_cent_bins_2)``.

        NB: the covariance is only computed between observables 
            not between sample points.

        `extra_std` is additional uncertainty which is added to each GP's
        predictive uncertainty, e.g. to account for model systematic error.
        It may either be a scalar or an array-like of length nsamples.

        """
        gp_mean = [gp.predict(X, return_cov=return_cov) for gp in self.gps]

        if return_cov:
            gp_mean, gp_cov = zip(*gp_mean)

        mean = self._inverse_transform(
            np.concatenate([m[:, np.newaxis] for m in gp_mean], axis=1)
        )

        if return_cov:
            # Build array of the GP predictive variances at each sample point.
            # shape: (nsamples, npc)
            gp_var = np.concatenate([
                c.diagonal()[:, np.newaxis] for c in gp_cov
            ], axis=1)

            # Add extra uncertainty to predictive variance.
            extra_std = np.array(extra_std, copy=False).reshape(-1, 1)
            gp_var += extra_std**2

            # Compute the covariance at each sample point using the
            # pre-calculated arrays (see constructor).
            cov = np.dot(gp_var, self._var_trans).reshape(
                X.shape[0], self.nobs, self.nobs
            )
            cov += self._cov_trunc

            return mean, cov
        else:
            return mean


    def sample_y(self, X, n_samples=1, random_state=None):
        """
        Sample model output at `X`.

        Returns a nested dict of observable arrays, each with shape
        ``(n_samples_X, n_samples, n_cent_bins)``.

        """
        # Sample the GP for each emulated PC.  The remaining components are
        # assumed to have a standard normal distribution.
        return self._inverse_transform(
            np.concatenate([
                gp.sample_y(
                    X, n_samples=n_samples, random_state=random_state
                )[:, :, np.newaxis]
                for gp in self.gps
            ] + [
                np.random.standard_normal(
                    (X.shape[0], n_samples, self.pca.n_components_ - self.npc)
                )
            ], axis=2)
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='train emulators with the model dataset',
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        '-par', '--parameter_file', type=str,
        help='model parameter filename')
    parser.add_argument(
        '-t', '--training_set_path', type=str,
        help='path for the training data set from model'
    )
    parser.add_argument(
        '--npc', type=int,
        help='number of principal components'
    )
    parser.add_argument(
        '--nrestarts', type=int,
        help='number of optimizer restarts'
    )

    parser.add_argument(
        '--retrain', action='store_true',
        help='retrain even if emulator is cached'
    )

    args = parser.parse_args()
    kwargs = vars(args)

    emu = Emulator(**kwargs)
    print('{} PCs explain {:.5f} of variance'.format(
        emu.npc,
        emu.pca.explained_variance_ratio_[:emu.npc].sum()
    ))

    for n, (evr, gp) in enumerate(zip(
            emu.pca.explained_variance_ratio_, emu.gps
    )):
        print(
            'GP {}: {:.5f} of variance, LML = {:.5g}, kernel: {}'
            .format(n, evr, gp.log_marginal_likelihood_value_, gp.kernel_)
        )
