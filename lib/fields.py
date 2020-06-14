import numpy.random as rnd
import numpy as np
import params as p


class HamiltonianFieldEvolution:
    def __init__(self, mu, gamma):
        self._mu = mu
        self.gamma = gamma

    def mu(self, t:float):
        return self._mu(t)


class AbstractControlField:

    def __init__(self, h: HamiltonianFieldEvolution, num_t: int,dt, alpha, parameter, randomize=False):
        rnd.seed(1814)
        self.H = h
        self.parameter = parameter
        self.alpha = alpha
        self.randomize = randomize
        self.num_t = num_t
        self.dt = dt
        # Xi
        self.values = np.zeros((num_t,)) if not randomize else rnd.rand(num_t)


class ControlField(AbstractControlField):
    @staticmethod
    def t_update( t_index):
        return t_index - 1

    def update(self, t_index, conjugate_control: AbstractControlField, chi: np.array, rho: np.array):
        t = t_index * self.dt
        t_index_update = self.t_update(t_index)

        part1 = (1 - self.parameter) * conjugate_control.values[t_index_update]
        part2 = - self.parameter / self.alpha * self.H.gamma * np.imag(np.vdot(chi[t_index_update],
                                                                               np.dot(self.H.mu(t),
                                                                                      rho[t_index])))
        self.values[t_index] = -(part1 + part2)


class ControlTildeField(ControlField):
    @staticmethod
    def t_update(t_index):
        return t_index


class ControlFields:
    def __init__(self, h: HamiltonianFieldEvolution, params: p.KrotovParams):
        self.eta = params.eta
        self.delta = params.delta
        self.alpha = params.alpha
        self.num_t = params.num_t
        dt = params.dt
        self.control = ControlField(h, self.num_t,dt, self.alpha, self.delta, randomize= True)
        self.control_tilde = ControlTildeField(h, self.num_t,dt, self.alpha, self.eta)


