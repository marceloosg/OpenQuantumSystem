import numpy.random as rnd
import numpy as np
import states
import params as p
import hamiltonians


class AbstractControlField:

    def __init__(self, h:object, num_t: int,dt, alpha, parameter, randomize=False):
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
    def update(self, t_index, conjugate_control: AbstractControlField, chi, rho):
        t = t_index * self.dt
        t_index_update = t_index - 1
        ex = self.values
        ex_tilde = conjugate_control.values

        #print("control \n\t t{} Ex{} \n".format(t, ex))
        #print("    \n\t Ex~{}\n".format(ex_tilde[t_update]))
        #print("\t tup{} \n\tchi {}\n\t mu{} \n\t rho{} \n".format(t_update, chi[t_update], self.H.mu(t), rho[t]))
        #print("\t p{} gamma{} alpha{}\n".format(self.parameter,self.H.gamma,self.alpha))

        part1 = (1 - self.parameter) * conjugate_control.values[t_index_update]
        part2 = - self.parameter / self.alpha * self.H.gamma * np.imag(np.vdot(chi[t_index_update],
                                                                               np.dot(self.H.mu(t),
                                                                                      rho[t_index])))
        #print("\t part1{} \n\t part2{}\n".format(part1,part2))
        self.values[t_index] = -(part1 + part2)


class ControlTildeField(AbstractControlField):
    def update(self, t_index, conjugate_control: AbstractControlField, chi, rho):
        t = t_index * self. dt
        t_index_update = t_index -1
        ex = conjugate_control.values
        ex_tilde = self.values
        tag = "tilde"

        #print("control {}\n\t t{} Ex{} \n".format(tag, t, ex))
        #print("    \n\t Ex~{}\n".format(ex_tilde[t_update]))
        #print("\t tup{} \n\tchi {}\n\t mu{} \n\t rho{} \n".format(t_update, chi[t_update], self.H.mu(t), rho[t]))
        #print("\t p{} \n\t gamma{}".format(self.parameter,self.H.gamma))
        part1 = (1 - self.parameter) * conjugate_control.values[t_index_update]
        part2 = - self.parameter / self.alpha * self.H.gamma * np.imag(np.vdot(chi[t_index_update], np.dot(self.H.mu(t),
                                                                                                     rho[t_index])))
        #print("\t part1{} \n\t part2{}\n".format(part1,part2))
        self.values[t_index] = -(part1 + part2)


class ControlFields:
    def __init__(self, h, params: p.KrotovParams):
        self.eta = params.eta
        self.delta = params.delta
        self.alpha = params.alpha
        self.num_t = params.num_t
        dt = params.dt
        self.control = ControlField(h, self.num_t,dt, self.alpha, self.delta, randomize= True)
        self.control_tilde = ControlTildeField(h, self.num_t,dt, self.alpha, self.eta)

    def update_control(self, t_index, chi, rho):
        self.control.update(t_index, self.control_tilde, chi, rho)

    def update_control_tilde(self, t_index, chi, rho):
        self.control_tilde.update(t_index, self.control, chi, rho)

    def evolve_control(self, chi: states.StateVector, rho: states.StateVector):
        # Evolve Control forwards
        time_steps = range(0, self.control.num_t - 1)
        #print("Control {}".format(time_steps))
        for t in time_steps:
            self.update_control(t, chi.states, rho.states)
            rho.forward_update(t, self.control)

    def evolve_control_tilde(self, chi: states.StateVector, rho: states.StateVector):
        # Evolve Control Tilde backwards, due to only having co-state at T for current iteration
        time_steps = range(self.control_tilde.num_t - 1, 0, -1)
        #print("Control tilde {}".format(time_steps))

        for t in time_steps:
#            print("{}".format(t))
            # Update Control Tilde using current time T for chi and rho
            self.update_control_tilde(t, chi.states, rho.states)
            # Update the next co-state at T-1 using the current value for Control Tilde
            chi.backward_update(t, self.control_tilde)
