import scipy.stats
import scipy.linalg
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import ode
import hamiltonians
import fields
import states
import params as p
import util


class Krotov:
    dt: float

    def __init__(self, h: hamiltonians.Hamiltonian, params: p.KrotovParams):
        # Hamiltonian
        self.H = h

        # Parameters for Krotov
        self.fields = fields.ControlFields(self.H, params)
        self.params = params

        # Density Matrix
        self.rho = states.StateVector(params.num_t, params.dt, self.H)
        self.chi = states.StateVector(params.num_t, params.dt, self.H)
        self.set_rho(params.rho)

        # calculate exp(iH(T)) tau exp(-iH(T))
        self.target_interaction = self.H.calculate_interation_picture_operator(params.rho_target, params.t_f)
        self.target_interaction = util.matrix2col(self.target_interaction)

        # Results
        self.cost = []

    # Sets initial rho
    def set_rho(self, array):
        self.rho.initial_state(array)

    # Calculates
    def op(self, psi):
        return np.vdot(self.target_interaction, psi) * self.target_interaction

    # Updates Psi and Control at the same time
    def evolution_psi(self):
        self.fields.evolve_control(self.chi, self.rho)
        # update the first control for the next iteration
        t = self.params.num_t - 1
        self.fields.update_control(t, self.chi.states, self.rho.states)

    # Updates Chi and Control_tilde at the same time
    def evolution_chi(self):
        self.fields.evolve_control_tilde(self.chi, self.rho)
        # update the first control for the next iteration
        t = 0
        self.fields.update_control_tilde(t, self.chi.states, self.rho.states)

    # Main Krotov loop
    def run_krotov(self, num_iter):
        # Set initial psi time evolution using initial random control
        #print("Ex {}".format(self.fields.control.values))
        self.rho.evolution_update(self.fields.control)
        #print("Psi {}".format(self.rho.states))
        for i in range(0, num_iter):
            print("Iteration : {} out of {}".format(i + 1, num_iter))
            # Set chi- costates for time T
            self.chi.operator_update(self.op, self.rho)
            #print("Chi {}".format(self.chi.states))
            # Update Chi (back propagate) and Control Field Tilde using current guess Control Field
            self.evolution_chi()
            #print("Chi {}".format(self.chi.states))
            # Update Psi and Control Field using current Chi(at t-1) and Control Field
            self.evolution_psi()
            #print("Psi {}".format(self.rho.states))
            # Calculate Cost
            self.cost.append(self.j_cost())

    def overlap_integral(self):
        t_index = self.params.num_t - 1
        ket = self.rho.states[t_index]
        bra = ket
        new_ket = self.op(ket)
        val = np.vdot(bra, new_ket)
        return val

    # Cost Function J
    def j_cost(self):
        xi = self.fields.control.values
        integration = scipy.integrate.simps(xi ** 2, x=self.params.t)
        cost = self.overlap_integral() - self.fields.alpha * integration
        return cost


if __name__ == '__main__':
    parameters = p.KrotovParams()
    H = hamiltonians.Hamiltonian(parameters)

    times_krotov = 50

    k = Krotov(H, params=parameters)  #
    k.run_krotov(times_krotov)  #

    plt.plot(range(times_krotov), np.array(k.cost), 'r')
    plt.savefig('mar-cost.png')
