import numpy as np

import scipy.signal


class SENMF(object):

    def __init__(self, n_bases, window_width, n_iter, X_shape, A_accel=1.0, D_accel=1.0, epsilon=None):
        """
        epsilon      if epsilon is set, fitting will stop when the difference
                     between 2 updates is less than epsilon, or n_iter has been exceeded
        """
        self.N_timesteps, self.N_features = X_shape
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.window_width = window_width
        self.n_bases = n_bases
        self.fit_iter = 0
        self.A_accel = A_accel
        self.D_accel = D_accel

    def rand_A(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.random((self.n_bases, self.N_timesteps))+2

    def ones_A(self):
        return np.ones((self.n_bases, self.N_timesteps))

    def rand_D(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.random((self.n_bases, self.window_width, self.N_features))+2

    def fit_error(self, X, A, D):
        return np.linalg.norm(self._residual(A, D, X))

    """
    Add a scale or accel param for A and D.
    accel 0.0 is effectively locking
    accel 1.0 is default
    anything in between slows down the fit
    """
    def fit(self, X, A, D):
        self.fit_iter = 0
        fit_err = self.fit_error(X, A, D)
        for _ in range(self.n_iter):
            self.fit_iter += 1
            if self.A_accel != 0.0:
                self._update_activations(A, D, X)
            if self.D_accel != 0.0:
                self._update_dictionary(A, D, X)
            n_fit_err = self.fit_error(X, A, D)
            if self.epsilon is not None and (fit_err - n_fit_err) < self.epsilon:
                break
            else:
                fit_err = n_fit_err

        return A, D

    def reconstruct(self, A, D):
        X_bar = np.zeros((self.N_timesteps, self.N_features))

        for basis, activation in zip(D, A):
            X_bar += scipy.signal.fftconvolve(basis.T, np.atleast_2d(activation)).T[:self.N_timesteps]

        return X_bar

    def _residual(self, A, D, X):
        return X / self.reconstruct(A, D)

    def _update_activations(self, A, D, X):
        for t_prime in range(self.window_width):
            R = self._residual(A, D, X)
            U_A = np.einsum("jk,tk->jt", D[:,t_prime,:]/np.atleast_2d(D[:,t_prime,:].sum(axis=1)).T, R[t_prime:])
            A[:,:-t_prime or None] *= self.A_accel * U_A

    def _update_dictionary(self, A, D, X):
        for t_prime in range(self.window_width):
            R = self._residual(A, D, X)
            U_D = np.einsum("jn,ni->ji", A[:,:-t_prime or None]/np.atleast_2d(A[:,:-t_prime or None].sum(axis=1)).T, R[t_prime:])
            D[:,t_prime,:] *= self.D_accel * U_D

