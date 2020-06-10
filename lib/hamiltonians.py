import numpy as np
import scipy
import params as p
import fields as f
import util


class Hamiltonian:
    def __init__(self, params: p.KrotovParams):
        # Sigma Matrices
        self.num_basis = 2
        self.gamma = params.gamma
        self.target = params.target
        self.n_bar, self.r_fp = params.calculate_beta_params()
        self.dt = params.dt
        self.omega = params.omega
        self.sigmaX = np.array([[0, 1.], [1., 0]], dtype='complex')
        self.sigmaY = np.array([[0, -1.j], [1.j, 0]], dtype='complex')
        self.sigmaZ = np.array([[1., 0], [0, -1.]], dtype='complex')
        self.I = np.array([[1., 0], [0, 1.]], dtype='complex')
        self.sigmaMinus = 0.5 * (self.sigmaX - 1.j * self.sigmaY)
        self.sigmaPlus = 0.5 * (self.sigmaX + 1.j * self.sigmaY)
        self.free_hamiltonian = self.omega / 2 * self.sigmaZ
        self._H_1 = self.calculate_lindbladian()

    def mu_prime(self, t:float):
        mu_prime = - (np.cos(self.omega * t) * self.sigmaX - np.sin(
            self.omega * t) * self.sigmaY)
        #mu_prime = self.calculate_interation_picture_operator(self.sigmaX, t)
        return mu_prime

    def mu(self, t:float):
        mu = np.kron(self.I, self.mu_prime(t)) - np.kron(np.conjugate(self.mu_prime(t)), self.I)
        return mu

    def h0(self, t:float, field: f.AbstractControlField):
        t_index = int(t / self.dt)
        return self.mu(t) * field.values[t_index]

    # L is Lindblad and C is a matrix
    def L(self, C):
        return (np.kron(np.conjugate(C), C) - 0.5 * (
            np.kron(self.I, np.dot(np.conjugate(np.transpose(C)), C))) - 0.5 * (
                    np.kron(np.dot(np.transpose(C), np.conjugate(C)), self.I)))

    # H_1 : Thermal Lindbladian in Liouville Form
    def calculate_lindbladian(self):
        return (self.L(np.sqrt(2 * (1 + self.n_bar)) * self.sigmaMinus) + self.L(
            np.sqrt(2 * self.n_bar) * self.sigmaPlus))

    # Final Liouville Space Hamiltonian
    def h(self, t:float, field: f.AbstractControlField):
        return -1.j * self.h0(t, field) + self.gamma * self._H_1

    def time_evolution_operator(self, t:float):
        # return exp(-i H_free t )
        kernel = 1.j * self.free_hamiltonian * t
        return scipy.linalg.expm(kernel)

    def calculate_interation_picture_operator(self, operator, t:float):
        # return U operator U_dagger
        u = self.time_evolution_operator(t)
        u_dagger = np.conjugate(u)
        matrix_target_interaction = np.dot(u,
                                           np.dot(operator,
                                                  u_dagger))
        return matrix_target_interaction
