import scipy
import numpy as np
from util import matrix2col
import fields as f


class AbstractStateVector:
    def __init__(self, num_t: int, dt: float, h_obj):
        self.num_t = num_t
        self.T = self.num_t - 1
        self.dt = dt
        self.states = np.zeros((self.num_t, h_obj.num_basis ** 2), dtype='complex')
        self.H = h_obj.h

    def evolution_update(self, field: object, forward=True):
        update = self.forward_update if forward else self.backward_update
        time_steps = range(0, self.num_t - 1) if forward else (self.num_t - 1, 0, -1)
        for t in time_steps:
            update(t, field)

    def initial_state(self, array):
        self.states[0, :] = matrix2col(array)

    def forward_update(self, t_index, field: object):
        t = t_index * self.dt
        # print("psi \n\t t{} Ex{} \n".format(t_index, field[t]))
        # print("\t H{} dt{} rhoi{} \n".format(self.H(t, field), self.dt, self.states[t]))
        column = np.dot(scipy.linalg.expm(self.H(t, field) * self.dt), self.states[t_index])
        self.states[t_index + 1] = column
        # print("\t rhof{}".format(column))

    def backward_update(self, t_index, field: object):
        t = t_index * self.dt
        # print("chi \n\t t{} Ex_tilde{} \n".format(t_index, field[t]))
        # print("\t H{} dt{} chii{} \n".format(self.H(t, field), self.dt, self.states[t]))
        u = scipy.linalg.expm(np.conjugate(np.transpose(-self.H(t, field))) * -self.dt)
        column = np.dot(u, self.states[t_index])
        self.states[t_index - 1] = column
        # print("\t chif{}".format(column))

    def apply_operator_at(self, op):
        return op(self.states[self.T])

    def apply_operator(self, op):
        return op(self.states)


class StateVector(AbstractStateVector):
    def operator_update(self, op, target_state: AbstractStateVector):
        self.states[self.num_t - 1] = target_state.apply_operator_at(op)
