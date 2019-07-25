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
    def __init__(self, training_set_path=".", npc=10, nrestarts=0):
        self._load_training_data(training_set_path)

        self.npc = npc
        nev, self.nobs = self.model_data.shape

        self.scaler = StandardScaler(copy=False)
        self.pca = PCA(copy=False, whiten=True, svd_solver='full')

        # Standardize observables and transform through PCA.  Use the first
        # `npc` components but save the full PC transformation for later.
        Z = self.pca.fit_transform(
                self.scaler.fit_transform(self.model_data))[:, :npc]

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
        self.model_data = []
        self.design_points = []
        for iev in glob(path.join(data_path, "*")):
            with open(path.join(iev, "sample.txt"), "r") as parfile:
                parameters = []
                for line in parfile:
                    line = line.split()
                    parameters.append(float(line[1]))
            self.design_points.append(parameters)
            temp_data = np.loadtxt(path.join(iev, "output.txt"))
            self.model_data.append(temp_data)
        self.design_points = np.array(self.design_points)
        self.model_data = np.array(self.model_data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='train emulators with the model dataset',
        argument_default=argparse.SUPPRESS
    )
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
