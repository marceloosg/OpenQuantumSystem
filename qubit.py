from typing import Optional

import scipy.stats
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import ode
from hamiltonian import Hamiltonian
from fields import ControlFields
from states import StateVector

'''Krotov algorithm for Open Quantum Systems'''

''' 

d |PSI >> = (-i H_0 -  L(C)) |PSI>>
-
dt

'''


class Krotov:
    dt: float

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
        self.rho = StateVector(num_t, self.dt, H)
        self.chi = StateVector(num_t, self.dt, H)

        # Time Independent Lindbladian
        self.target_interaction = self.H.apply_time_evolution_operator(target, self.t[-1])

    # Sets initial rho
    def set_rho(self, array):
        self.rho.initial_state(array)

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

    def evolution_Psi(self, string='not initial'):
        self.fields.evolve_control(self.chi,self.rho)
        self.rho.evolution_update()
        T = self.num_t - 1
        self.fields.update_control(T, self.chi, self.rho)

    def evolution_Chi(self):
        self.fields.evolve_control_tilde(self.chi, self.rho)
        self.chi.evolution_update(forward=False)
        t = 0
        self.fields.update_control_tilde(t, self.chi, self.rho)

    def Run_Krotov(self, num_iter):
        T = self.num_t - 1
        self.rho.evolution_update()

        self.cost = []
        self.oint = []

        for i in range(0, num_iter):
            print("Iteration : {} out of {}".format(i + 1, num_iter))
            self.chi.operator_update(self.O, self.rho)
            self.evolution_Chi()
            self.evolution_Psi()
            self.cost.append(self.J())

    # Returns <<PSI|O|PSI>>>
    def Overlap(self, psi):
        return (np.vdot(psi, self.O(psi)))

    def Overlap_Integral(self):
        T_index = self.num_t - 1
        ket = self.rho.states[T_index]
        bra = ket
        newKet = self.O(ket)
        val = np.vdot(bra, newKet)
        return val

    # Cost Function J
    def J(self):
        xi = self.fields.control.values
        integration = scipy.integrate.simps(xi ** 2, x=self.t)
        J = self.Overlap_Integral() - self.fields.alpha * integration
        return J


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
