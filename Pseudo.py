import numpy as np

# Main call

# Initialize (kappa, n_bar, w1, lamda, t_i, t_f, N_iterations, target)
# kappa =
# n_bar =
# lambda
# w = qubit gap , unit h_bar =1
# t_i = initial time
# t_f = end time ,  (we choose to fix the time instead of constraining the hamiltonian)

# N_iterations, number of evolution iterations for each Kroton time-step
# k_max, number of krotov iteractions

# alpha , algorithm parameter, definies the relative importance between fluence/fidelity in the cost function
# eta, algorithm parameter (from ansatz of self-consistent set of equations)
# delta, algorithm parameter (from ansatz of self-consistent set of equations)
# epsilon, time-dependent control field
# gamma, controls the rate of decoherence

# initial_rho
# target , thermalized final state
# s , speed up factor based on T = T free/s

params = {"gamma": 0.1,
          "eta": 1.5,
          "delta": 1.5,
          "alpha": 0.0001,
          "epsilon": 0.1}

krotov_instance = Krotov(w=2,
                         t_i=0,
                         N_iterations=100,
                         s=2,
                         params=params,
                         initial_rho=[[(0.5, 0j), (0, 0.19j)], [(0, -0.19j), (0.5, 0j)]],
                         target=[[0.4, 0], [0, 0.6]])  #

krotov_instance.Solve(k_max=5)


# Class definition

class Krotov:

    def initialize_Hamiltonian(self):
        pass

    def initilize_Psi(self):
        # propagate |psi> according to
        # n is a index from 0 to N_iteractions
        # |rho(t_(n+1)) > exp(-H(t)dt) | rho(t_n) >
        pass

    def initialize_reverse_time_index(self):
        # time index, based on the number of N_iterations
        # the indexs run reversed from N_iterations -1 to 0 to apply backpropagation of operators
        pass

    def initialize_Xi(self):
        # for all time index set Xi at a random configuration
        pass

    def init(self, w, t_i, N_iterations, param, initial_rho, target):
        self.w = w
        self.t_i = t_i
        self.N_iterations = N_iterations
        self.param = param
        self.initial_rho = initial_rho
        self.target = target
        self.Hamiltonian = self.initialize_Hamiltonian()
        self.reverse_time_index = self.initialize_reverse_time_index()

    def mu_prime(self, t_index):
        #  mu = - cos(wt)sigmaX - sin(wt)sigmaY ?  -> outputed from  [ w Sz , Xi(t) Sx ]
        # I am not sure.
        pass


    def mu(self, t_index):
        # evolve mu according to Eq 14. For unitary evolution it reduces to I X mu_prime - mu_prime_dagger X I
        pass

    def update_Xi_tilde(self, t_index):
        # evolve Xi_tilde according to Eq.21 in the reverse direction (complex conjulgated values)
        # Xi_tilde(t) = (1 - eta) * Xi(t)_dagger - eta/alpha  < chi (t) | mu(t) | rho (t) >
        pass

    def update_Chi(self, t_index):
        # Apply reversed time evolution operator H_tilde on Chi, see A_tilde(t) on eq. 20
        # | Chi[t-1] > = exp(-H_tilde_dagger(t) dt ) | Chi >
        pass

    def update_Psi(self, t_index):
        # Apply forward time evolution operator H(t) on psi, , see A(t) on eq. 19
        # | psi(t+1) > = exp(H(t)dt) | psi (t) >
        pass


    def evolution_Chi(self):
        for t in self.reverse_time_index:
            self.update_Xi_tilde(t)
            self.update_Chi(t)
        t = 0
        self.update_Xi_tilde(t)

    def Solve(self, k_max):
        self.initialize_Psi()
        self.initialize_Xi()

        # for each Krotov iteration
        for k in range(0, k_max):
            self.evolution_Chi()
            self.evolution_Psi()
