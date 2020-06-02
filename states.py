from hamiltonian import Hamiltonian,np
import scipy


class StateVector:
    def __init__(self, num_t, dt,  H:Hamiltonian):
        self.num_t = num_t
        self.dt = dt
        self.states = np.zeros((self.num_t, self.H.num_basis ** 2), dtype='complex')
        self.forward_op = self.H.h
        self.backward_op = self.H.h_tilde

    def forward_update(self, t_index):
        t = t_index
        column = np.dot(scipy.linalg.expm(self.forward_op(t) * self.dt), self.states[t])
        self.states[t + 1] = column
        
    def backward_update(self, t_index):
        t = t_index
        column = np.dot(scipy.linalg.expm(np.conjugate(np.transpose(-self.backward_op(t))) * -self.dt), self.states[t])
        self.states[t - 1] = column
