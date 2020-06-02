import numpy.random as rnd
import numpy as np
from hamiltonian import Hamiltonian


class ControlField:
    def __init__(self,H:Hamiltonian, num_t, alpha, parameter, back_propagate=False):
        self.H = H
        self.parameter = parameter
        self.alpha = alpha
        self.back_propagate = back_propagate
        # Xi
        self.control = np.zeros((num_t,)) if back_propagate else rnd.rand(num_t)

    def update(self, t_index, conjugate_control, chi, rho):
        t = t_index
        t_update = t - 1 if self.back_propagate else t
        part1 = (1 - self.parameter) * conjugate_control[t_update]
        part2 = - self.H.gamma * np.imag(np.vdot(chi[t_update], np.dot(self.H.mu(t), rho[t])))
        self.control[t] = -(part1 + self.parameter/self.alpha * part2)


class ControlFields:
    def __init__(self,num_t, H:Hamiltonian, eta=1.5, delta=1.5, alpha=0.001):
        self.eta =eta
        self.delta= delta
        self.alpha = alpha
        self.control = ControlField(H, num_t,alpha, delta)
        self.control_tilde = ControlField(H, num_t, alpha, delta,back_propagate=True)


    def update_control(self, t_index, chi,rho):
        self.control.update(t_index, self.control_tilde, chi,rho)

    def update_control_tilde(self, t_index, chi,rho):
        self.control_tilde.update(t_index, self.control, chi,rho)
