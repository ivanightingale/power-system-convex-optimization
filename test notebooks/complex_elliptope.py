import numpy as np
from numpy import linalg as la
from numpy import random as rnd
from scipy.linalg import expm
from scipy.linalg import solve_continuous_lyapunov as lyap

from pymanopt.manifolds.manifold import (
    EuclideanEmbeddedSubmanifold,
    Manifold,
    RetrAsExpMixin,
)
from pymanopt.tools.multi import multilog, multiprod, multisym, multitransp


class ComplexElliptope(Manifold, RetrAsExpMixin):
    # RetrAsExpMixin: use when exponential map for the manifold is not available. exp() will return retr() instead
    # X = YY* with only full-rank Y (equivalence class under orthogonal group)

    def __init__(self, n, k):
        self._n = n
        self._k = k

        name = (
            f"Quotient manifold of {n}x{n} complex psd matrices of rank {k} "
            "with unit diagonal elements"
        )
        dimension = int(n * (k - 1) - k * (k - 1) / 2)
        super().__init__(name, dimension)

    @property
    # Returns the scale of the manifold. Used by the trust-regions solver to determine default initial and maximal
    # trust-region radii.
    def typicaldist(self):
        return 10 * self._k

    # inner product of tangent vectors U and V at Y
    def inner(self, Y, U, V):
        return np.tensordot(U.conj(), V)  # FIXME: why is only the real part used in complex_circle and complex_grassmann?

    # norm of tangent vector U at Y
    def norm(self, Y, U):
        return np.sqrt((self.inner(Y, U, U)).real)

    # projection of H onto the horizontal space at Y (a subspace of the tangent space at Y which represents the
    # tangent space under quotient)
    def proj(self, Y, H):
        # project H to tangent space at Y
        eta = self._project_rows(Y, H)

        # project onto the horizontal space at Y
        YtY = Y.conj().T @ Y
        AS = Y.conj().T @ eta - eta.conj().T @ Y
        # find skew-Hermitian matrix Omega which solves the Sylvester equation
        Omega = lyap(YtY, AS)
        return eta - Y @ (Omega - Omega.conj().T) / 2

    # retraction of vector U at Y
    def retr(self, Y, U):
        return self._normalize_rows(Y + U)

    # Euclidean gradient to Riemannian gradient conversion. We only need the
    # ambient space projection: the remainder of the projection function is not
    # necessary because the Euclidean gradient must already be orthogonal to
    # the vertical space.
    def egrad2rgrad(self, Y, egrad):
        return self._project_rows(Y, egrad)

    # Converts the Euclidean to the Riemannian Hessian.
    def ehess2rhess(self, Y, egrad, ehess, U):
        scaling_grad = (egrad * Y).sum(axis=1)
        hess = ehess - U * scaling_grad[:, np.newaxis]

        scaling_hess = (U * egrad + Y * ehess).sum(axis=1)
        hess -= Y * scaling_hess[:, np.newaxis]

        return self.proj(Y, hess)

    # a random point on the manifold
    def rand(self):
        return self._normalize_rows(rnd.randn(self._n, self._k) + rnd.randn(self._n, self._k) * 1j)

    # a random vector in the horizontal space at Y
    def randvec(self, Y):
        H = self.proj(Y, self.rand())
        return H / self.norm(Y, H)  # TODO: why?

    # transport U from the tangent space at Y to the tangent space at Z
    def transp(self, Y, Z, U):
        return self.proj(Z, U)

    # copy of Y where rows are l2-normalized
    def _normalize_rows(self, Y):
        return Y / la.norm(Y, axis=1)[:, np.newaxis]

    # orthogonal projection of each row of H to the tangent space at the corresponding row of Y, seen as a point on a
    # sphere.
    def _project_rows(self, Y, H):
        # compute the projection of each row of H onto the corresponding row of Y, which is H[i, :].T @ Y[i, :]
        # note that the norm of each row of Y is 1, so no additional normalization needed
        HprojY = Y * (H * Y.conj()).sum(axis=1)[:, np.newaxis]
        # project H to the tangent space
        return H - HprojY

    # zero vector in the tangent space of Y
    def zerovec(self, Y):
        return np.zeros((self._n, self._k))