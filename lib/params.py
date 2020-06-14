import numpy as np
import util


class KrotovParams:
    def __init__(self, eta=1.5, delta=1.5, alpha=0.001, epsilon=0.1, gamma=0.1, omega=2.,
                 rho_init=np.array([[0.5, -0.19j], [0.19j, 0.5]], dtype='complex'),
                 rho_target=np.array([[0.4, 0], [0, 0.6]], dtype='complex'),
                 num_t=2000, speedup_factor=2):

        self.eta = eta
        self.delta = delta
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.omega = omega
        self.speedup_factor = speedup_factor

        #rho
        self.rho = rho_init
        self.rho_target = rho_target
        self.target = util.matrix2col(rho_target)

        # Thermal
        self.n_bar, self.r_fp = self.calculate_beta_params()

        # Time
        self.t_i = 0
        self.t_f = self.t_final()
        self.num_t = num_t
        self.t = np.linspace(self.t_i, self.t_f, self.num_t)
        self.dt = self.t[1] - self.t[0]

    def calculate_beta_params(self):
        # p00 = gnd = target[3] = 1/(exp(-wB)+1)
        #
        # exp(-wB) = 1/gnd-1 = (1-gnd)/gnd
        # exp(wB) = gnd/(1-gnd)
        # exp(wb) +1 = 1/(1-gnd)
        # exp(wB) -1 = (2gnd-1)/(1-gnd)
        # nbar = 1/(exp(wB)-1) = (1-gnd)/(2gnd - 1)
        gnd = self.target[3]
        assert np.real(self.target[3]+self.target[0]) == 1, "Density matrix sanity check"
        assert np.real(gnd) >= 0.5, "gnd < 0.5 implies negative fixed point for the density matrix"
        assert np.real(gnd) > 0.5, "gnd({}) == 0.5 implies null fixed point, null beta, infinite temperature".format(gnd)
        y = gnd / (1.0  - gnd)
        nbar = 1.0 / (y - 1)
        r_fp = (y - 1) / (y + 1)
        #nbar = (1-gnd)/(2*gnd-1)
        #r_fp = 2*gnd-1
        return nbar, r_fp

    def calculate_t_epsilon_free(self):
        rho0 = self.rho
        I = np.array([[1, 0], [0, 1]])
        r = 2 * rho0 - I
        rz_0 = r[0, 0]
        rx_0 = np.real(r[0, 1])
        ry_0 = np.imag(r[0, 1])

        alpha = rx_0 ** 2 + ry_0 ** 2
        beta = rz_0 + self.r_fp
        x=(-alpha +np.sqrt(alpha ** 2 + 16 * self.epsilon **2 * beta ** 2) )/ (2 * beta ** 2)
        t_e_free = -np.log(x)/ (2*self.gamma)

        return t_e_free

    def t_final(self):
        t_e_free = np.abs(self.calculate_t_epsilon_free())
        return t_e_free / self.speedup_factor

