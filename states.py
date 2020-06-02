from hamiltonian import Hamiltonian, np
import scipy


class AbstractStateVector:
    def __init__(self, num_t: int, dt: float, H: Hamiltonian) -> object:
        self.num_t = num_t
        self.T = self.num_t - 1
        self.dt = dt
        self.states = np.zeros((self.num_t, self.H.num_basis ** 2), dtype='complex')
        self.forward_op = self.H.h
        self.backward_op = self.H.h_tilde

    def evolution_update(self, forward = True):
        update = self.forward_update if forward else self.backward_update
        time_steps = range(0, self.num_t - 1) if forward else (self.num_t - 1, 0, -1)
        for t in time_steps:
            update(t)

    def initial_state(self, array):
        self.states[0, :] = array

    def forward_update(self, t_index):
        t = t_index
        column = np.dot(scipy.linalg.expm(self.forward_op(t) * self.dt), self.states[t])
        self.states[t + 1] = column

    def backward_update(self, t_index):
        t = t_index
        column = np.dot(scipy.linalg.expm(np.conjugate(np.transpose(-self.backward_op(t))) * -self.dt), self.states[t])
        self.states[t - 1] = column

    def apply_operator_at(self, O):
        return O(self.states[self.T])

    def apply_operator(self, O):
        return O(self.states)



class StateVector(AbstractStateVector):
    def operator_update(self, O, target_state:AbstractStateVector):
        self.states[self.num_t-1] = target_state.apply_operator_at(O)




