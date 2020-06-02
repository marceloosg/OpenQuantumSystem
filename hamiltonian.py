import numpy as np
import scipy
import util


class Hamiltonian:
    def __init__(self, gamma, omega, n_bar):
        # Sigma Matrices
        self.num_basis = 2
        self.gamma = gamma
        self.n_bar = n_bar
        self.omega = omega
        self.sigmaX = np.array([[0, 1.], [1., 0]], dtype='complex')
        self.sigmaY = np.array([[0, -1.j], [1.j, 0]], dtype='complex')
        self.sigmaZ = np.array([[1., 0], [0, -1.]], dtype='complex')
        self.I = np.array([[1., 0], [0, 1.]], dtype='complex')
        self.sigmaMinus = 0.5 * (self.sigmaX - 1.j * self.sigmaY)
        self.sigmaPlus = 0.5 * (self.sigmaX + 1.j * self.sigmaY)
        self._H_1 = self.calculate_lindbladian()

    def mu_prime(self, t_index):
        mu_prime = self.sigmaX
        return mu_prime

    def mu(self, t_index):
        mu = np.kron(self.I, self.mu_prime(t_index)) - np.kron(np.conjugate(self.mu_prime(t_index)), self.I)
        return mu

    def h0(self, t_index, field):
        return self.mu(t_index) * field[t_index]

    def h0_tilde(self, t_index,field):
        return self.mu(t_index) * field[t_index]

    # L is Lindblad and C is a matrix
    def L(self, C):
        return (np.kron(np.conjugate(C), C) - 0.5 * (
            np.kron(self.I, np.dot(np.conjugate(np.transpose(C)), C))) - 0.5 * (
                    np.kron(np.dot(np.transpose(C), np.conjugate(C)), self.I)))

    # H_1 : Thermal Lindbladian in Liouville Form
    def calculate_lindbladian(self):
        return (self.L(np.sqrt(2 *  (1 + self.n_bar)) * self.sigmaMinus) + self.L(
            np.sqrt(2 * self.n_bar) * self.sigmaPlus))

    # Final Liouville Space Hamiltonian
    def h(self, t_index):
        return -1.j * self.h0(t_index) + self.gamma * self._H_1

    def h_tilde(self, t_index):
        return -1.j * self.h0_tilde(t_index) + self.gamma ** 2 * self._H_1

    def apply_time_evolution_operator(self, target, t):
        # Thermal or Target State
        omega_z_t=1.j * 0.5 * self.omega * self.sigmaZ * t
        matrix_target_interaction = np.dot(scipy.linalg.expm(omega_z_t,
                                                np.dot(util.col2matrix(target),
                                                       scipy.linalg.expm(-omega_z_t))))
        return util.matrix2col(matrix_target_interaction)

    def calculate_t_epsilon_free(self, epsilon, initial_rho):
        r_fp = 1 / (2 * self.n_bar + 1)
        gamma2 = self.gamma / r_fp
        rho0 = initial_rho
        I = np.array([[1, 0], [0, 1]])

        r = 2 * rho0 - I
        rz_0 = r[0, 0]
        rx_0 = np.real(r[0, 1])
        ry_0 = np.imag(r[0, 1])

        alpha = rx_0 ** 2 + ry_0 ** 2
        beta = rz_0 + r_fp

        T_e_free = np.log((-alpha ** 2 + np.sqrt(alpha ** 4 - 16 * epsilon * beta ** 2)) / (2 * beta ** 2) / (-gamma2))

        return (T_e_free)
