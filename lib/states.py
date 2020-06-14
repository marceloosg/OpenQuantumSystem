
import numpy as np
from util import matrix2col
import fields as f


class AbstractStateVector:
    def __init__(self, num_t: float, dt: float, num_basis: int):
        self.num_t = num_t
        self.T = self.num_t - 1
        self.dt = dt
        self.states = np.zeros((self.num_t, num_basis ** 2), dtype='complex')

    def apply_operator_at(self, op):
        return op(self.states[self.T])

    def apply_operator(self, op):
        return op(self.states)


class StateVector(AbstractStateVector):
    def operator_update(self, op, target_state: AbstractStateVector):
        self.states[self.T] = target_state.apply_operator_at(op)
