import numpy.random as rnd
import numpy as np
from hamiltonian import Hamiltonian


class AbstractControlField:
    def __init__(self, h: Hamiltonian, num_t, alpha, parameter, back_propagate=False):
        self.H = h
        self.parameter = parameter
        self.alpha = alpha
        self.back_propagate = back_propagate
        self.num_t = num_t
        # Xi
        self.values = np.zeros((num_t,)) if back_propagate else rnd.rand(num_t)


class ControlField(AbstractControlField):
    def update(self, t_index, conjugate_control: AbstractControlField, chi, rho):
        t = t_index
        t_update = t - 1 if self.back_propagate else t
        part1 = (1 - self.parameter) * conjugate_control[t_update]
        part2 = - self.H.gamma * np.imag(np.vdot(chi[t_update], np.dot(self.H.mu(t), rho[t])))
        self.values[t] = -(part1 + self.parameter/self.alpha * part2)


class ControlFields:
    def __init__(self,num_t, h: Hamiltonian, eta=1.5, delta=1.5, alpha=0.001):
        self.eta = eta
        self.delta = delta
        self.alpha = alpha
        self.control = ControlField(h, num_t,alpha, delta)
        self.control_tilde = ControlField(h, num_t, alpha, delta, back_propagate=True)

    def update_control(self, t_index, chi,rho):
        self.control.update(t_index, self.control_tilde, chi,rho)

    def update_control_tilde(self, t_index, chi,rho):
        self.control_tilde.update(t_index, self.control, chi,rho)

    def evolve_control(self, chi, rho):
        time_steps = range(0, self.control.num_t - 1)
        for t in time_steps:
            self.update_control(t,chi,rho)

    def evolve_control_tilde(self, chi, rho):
        time_steps = (self.control_tilde.num_t - 1, 0, -1)
        for t in time_steps:
            self.update_control_tilde(t,chi,rho)
