import numpy as np
import scipy
import params as p


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
        self.identity_matrix = np.array([[1., 0], [0, 1.]], dtype='complex')
        self.sigmaMinus = 0.5 * (self.sigmaX - 1.j * self.sigmaY)
        self.sigmaPlus = 0.5 * (self.sigmaX + 1.j * self.sigmaY)
        self.free_hamiltonian = self.omega / 2 * self.sigmaZ
        self._H_1 = self.calculate_lindbladian()

    def field_evolution_parameters(self):
        return self.mu, self.gamma

    def mu_prime(self, t: float):
        # Dinamical mapping of mu -> mu', Interaction Picture
        # exp(iH0t) mu exp(-iH0t) = - (cos(wt) SigmaX - sin(wt) SimgaY)
        # -> calculated explicitly to save computational effort
        mu_prime = - (np.cos(self.omega * t) * self.sigmaX - np.sin(
            self.omega * t) * self.sigmaY)
        # mu_prime = self.calculate_interation_picture_operator(self.sigmaX, t)
        return mu_prime

    def mu(self, t: float):
        # Apply Commutator
        mu = np.kron(self.identity_matrix, self.mu_prime(t)) -\
             np.kron(np.conjugate(self.mu_prime(t)), self.identity_matrix)
        return mu

    # lindblad is Lindblad and C is a matrix
    def lindblad(self, matrix):
        return (np.kron(np.conjugate(matrix), matrix) - 0.5 * (
            np.kron(self.identity_matrix, np.dot(np.conjugate(np.transpose(matrix)), matrix))) - 0.5 * (
                    np.kron(np.dot(np.transpose(matrix), np.conjugate(matrix)), self.identity_matrix)))

    # H_1 : Thermal Lindbladian in Liouville Form
    def calculate_lindbladian(self):
        return self.lindblad(np.sqrt(2 * (1 + self.n_bar)) * self.sigmaMinus) + self.lindblad(
            np.sqrt(2 * self.n_bar) * self.sigmaPlus)

    def time_evolution_operator(self, t: float):
        # return exp(-i H_free t )
        kernel = 1.j * self.free_hamiltonian * t
        return scipy.linalg.expm(kernel)

    def calculate_interation_picture_operator(self, operator, t: float):
        # return U operator U_dagger
        u = self.time_evolution_operator(t)
        u_dagger = np.conjugate(u)
        matrix_target_interaction = np.dot(u,
                                           np.dot(operator,
                                                  u_dagger))
        return matrix_target_interaction
