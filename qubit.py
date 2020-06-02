import scipy.stats
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import ode
from hamiltonian import Hamiltonian
from fields import ControlFields
import util


'''Krotov algorithm for Open Quantum Systems'''

''' 

d |PSI >> = (-i H_0 -  L(C)) |PSI>>
-
dt

'''


class Krotov:
    def __init__(self, H:Hamiltonian, t_i, t_f, num_t, target):

        # Hamiltonian
        self.H = H
        self.t_index = 0

        # Time
        self.t_i = t_i
        self.t_f = t_f
        self.num_t = num_t
        self.t = np.linspace(self.t_i, self.t_f, self.num_t)
        self.dt = self.t[1] - self.t[0]

        # Parameters for Krotov
        self.fields = ControlFields(num_t, H)

        # Density Matrix
        self.col_rho = np.zeros((self.num_t, self.H.num_basis ** 2), dtype='complex')
        self.col_chi = np.zeros((self.num_t, self.H.num_basis ** 2), dtype='complex')

        # Time Independent Lindbladian
        self.target_interaction = self.H.apply_time_evolution_operator(target, self.t[-1])

    # Sets initial rho
    def set_rho(self, array):
        self.col_rho[0] = array

    '''
	The equation becomes 
	d |PSI >> = H(t) |PSI>>
	-
	dt

	We solve it by exponentiation!

	|PSI (t+dt)>> = exp(H(t) dt) |Psi(t)>>
	'''

    # Overlap operator with target state O|PSI>>

    def O(self, psi):
        return (np.vdot(self.target_interaction, psi) * self.target_interaction)

    # Returns <<PSI|O|PSI>>>
    def Overlap(self, psi):
        return (np.vdot(psi, self.O(psi)))

    def update_Epsilon(self, t_index):
        t = t_index
        part1 = (1 - self.delta) * self.Ex_tilde[t - 1]
        part2 = -self.delta * self.gamma * np.imag(
            np.vdot(self.col_chi[t - 1], np.dot(self.mu(t), self.col_rho[t]))) / self.alpha
        self.Ex[t] = -(part1 + part2)

    def update_Epsilon_tilde(self, t_index):
        t = t_index
        part1 = (1 - self.eta) * self.Ex[t]
        part2 = -self.eta * self.gamma * np.imag(
            np.vdot(self.col_chi[t], np.dot(self.mu(t), self.col_rho[t]))) / self.alpha
        self.Ex_tilde[t] = -(part1 + part2)

    def evolution_Psi(self, string='not initial'):
        if string == 'initial':
            for t in range(0, self.num_t - 1):
                self.update_Psi(t)
        else:
            for t in range(0, self.num_t - 1):
                self.update_Epsilon(t)
                self.update_Psi(t)
            t = self.num_t - 1
            self.update_Epsilon(t)

    def evolution_Chi(self):
        for t in range(self.num_t - 1, 0, -1):
            self.update_Epsilon_tilde(t)
            self.update_Chi(t)
        t = 0
        self.update_Epsilon_tilde(t)

    def Run_Krotov(self, num_iter):
        T = self.num_t - 1
        self.evolution_Psi('initial')

        self.overlap = []
        self.cost = []
        self.oint = []

        for i in range(0, num_iter):
            print("Iteration : {} out of {}".format(i + 1, num_iter))
            self.col_chi[T] = self.O(self.col_rho[T])
            self.evolution_Chi()
            self.evolution_Psi()
            self.overlap.append(self.Overlap(self.col_rho[T]))
            self.oint.append(self.Overlap_Integral())
            self.cost.append(self.J())


    def Overlap_Integral(self):
        T_index = self.num_t - 1

        ket = self.col_rho[T_index]
        bra = ket
        newKet = self.O(ket)

        val = np.vdot(bra, newKet)

        return (val)

    # Cost Function J
    def J(self):
        integration = scipy.integrate.simps(self.Ex ** 2, x=self.t, dx=self.dt)

        J = self.alpha * integration

        return (J)




def T_final(T_e_free, s):
    return (T_e_free / s)


if __name__ == '__main__':

    n_bar = 1.  # Being FLOAT is VERY IMPORTANT
    omega = 2
    gamma = 1
    H = Hamiltonian(gamma,omega,n_bar)

    target = np.array([0.4, 0, 0, 0.6], dtype='complex')
    initial_rho = np.array([[0.5, -0.19j], [0.19j, 0.5]], dtype='complex')
    epsilon = .01

    T_e_free = np.abs(H.calculate_t_epsilon_free(epsilon, initial_rho))
    _s = 2.
    t_f = T_e_free / _s


    ''' Krotov With Control '''  ######################################
    N_iterations = 200
    times_krotov = 50


    k = Krotov(H, 0, t_f, N_iterations, target)  #
    k.set_rho_init(k.matrix2col(initial_rho))  #
    k.set_Ex("random")  #
    k.set_Para(eta=1.5, delta=1.5, alpha=0.001)  #
    k.Run_Krotov(times_krotov)  #
    #################################################################


    plt.plot(range(times_krotov), np.array(k.cost), 'r')
    plt.savefig('mar-cost.png')
# WITH    KROTOV
