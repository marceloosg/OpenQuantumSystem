import states
import scipy
import numpy as np
from util import matrix2col
from states import StateVector, AbstractStateVector
from params import KrotovParams
from fields import AbstractControlField, ControlFields, HamiltonianFieldEvolution
from hamiltonians import Hamiltonian


class LiouvilleHamiltonian(Hamiltonian):
    # Control Hamiltonian (Time dependent part)
    def h0(self, t: float, field: AbstractControlField):
        t_index = int(t / self.dt)
        return self.mu(t) * field.values[t_index]

    # Final Liouville Space Hamiltonian
    def h(self, t: float, field: AbstractControlField):
        return -1.j * self.h0(t, field) + self.gamma * self._H_1


class StateController(StateVector):
    def __init__(self, hamiltonian: LiouvilleHamiltonian, params: KrotovParams):
        super().__init__(params.num_t, params.dt, hamiltonian.num_basis)
        self.H = hamiltonian.h
        # calculate exp(iH(T)) tau exp(-iH(T))
        self.target_interaction = hamiltonian.calculate_interation_picture_operator(params.rho_target, params.t_f)
        self.target_interaction = matrix2col(self.target_interaction)

    # Calculates
    def op(self, psi):
        return np.vdot(self.target_interaction, psi) * self.target_interaction

    def apply_target_interaction(self, target: AbstractStateVector):
        self.operator_update(self.op, target)

    def apply_interaction_at_last_t(self):
        return self.apply_operator_at(self.op)

    def last_state(self):
        return self.states[self.T]

    def initial_state(self, array):
        self.states[0, :] = matrix2col(array)

    def forward_update(self, t_index, field: AbstractControlField):
        t = t_index * self.dt
        column = np.dot(scipy.linalg.expm(self.H(t, field) * self.dt), self.states[t_index])
        self.states[t_index + 1] = column
        # print("\t rhof{}".format(column))

    def backward_update(self, t_index, field: AbstractControlField):
        t = t_index * self.dt
        u = scipy.linalg.expm(np.conjugate(np.transpose(-self.H(t, field))) * -self.dt)
        column = np.dot(u, self.states[t_index])
        self.states[t_index - 1] = column
        # print("\t chif{}".format(column))

    def evolution_update(self, field: AbstractControlField, forward=True):
        update = self.forward_update if forward else self.backward_update
        time_steps = range(0, self.num_t - 1) if forward else (self.num_t - 1, 0, -1)
        for t in time_steps:
            update(t, field)


class FieldController(ControlFields):
    def __init__(self, h: Hamiltonian, p: KrotovParams):
        mu, gamma = h.field_evolution_parameters()
        super().__init__(HamiltonianFieldEvolution(mu,gamma), p)

    def update_control(self, t_index, chi: np.array, rho: np.array):
        self.control.update(t_index, self.control_tilde, chi, rho)

    def update_control_tilde(self, t_index, chi: np.array, rho: np.array):
        self.control_tilde.update(t_index, self.control, chi, rho)

    def evolve_control(self, chi: StateController, rho: StateController):
        # Evolve Control forwards
        time_steps = range(0, self.control.num_t - 1)

        for t in time_steps:
            self.update_control(t, chi.states, rho.states)
            rho.forward_update(t, self.control)

    def evolve_control_tilde(self, chi: StateController, rho: StateController):
        # Evolve Control Tilde backwards, due to only having co-state at T for current iteration
        time_steps = range(self.control_tilde.num_t - 1, 0, -1)

        for t in time_steps:
            # Update Control Tilde using current time T for chi and rho
            self.update_control_tilde(t, chi.states, rho.states)
            # Update the next co-state at T-1 using the current value for Control Tilde
            chi.backward_update(t, self.control_tilde)



