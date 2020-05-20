# -*- coding: utf-8 -*-

from enum import Enum  # For the different phases
import numpy as np
from scipy.optimize import linprog
from itertools import combinations
from math import log

eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]
def klBern(x, y):
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))
klBern_vect = np.vectorize(klBern)


#: Different phases during the OSSB algorithm
Phase = Enum('Phase', ['initialisation', 'exploitation', 'estimation', 'exploration'])

#: Default value for the :math:`\gamma` parameter, 0.0 is a safe default.
GAMMA = 0.0

##
def estimate_Lipschitz_constant(thetas):
    idx = [i for i in range(thetas.size)]
    combi = combinations(idx, 2)
    combi = list(combi)
    L_values = []
    for i,j in combi:
        L_values.append(abs(thetas[i]-thetas[j])/(abs(i-j)/thetas.size))
    return np.amax(L_values)
##
def solve_optimization_problem__Lipschitz(thetas, L=-1):
    if L==-1:
        tol = 1e-10
    else:
        tol = 1e-8

    theta_max = np.amax(thetas)
    c = theta_max - thetas  # c : (\theta^*-theta_k)_{k\in K}
    
    sub_arms = (np.nonzero(c))[0]
    opt_arms = (np.where(c==0))[0]

    ##
    if sub_arms.size==0:    # ex) arms' mean => all 0
        print("*****all arms are estimated as an optimal arm*****")
        # return np.zeros_like(thetas)
        return np.full(thetas.size, np.inf)

    # for unknown Lipschitz constant
    if L==-1:
        L = estimate_Lipschitz_constant(thetas)

    def get_confusing_bandit(k):
        # values : \lambda_i^k
        values = np.zeros_like(thetas)
        for i, theta in enumerate(thetas):
            values[i] = max(theta, theta_max-L*abs(k-i)/thetas.size)
        return values

    A_ub=np.zeros((sub_arms.size, sub_arms.size))
    for j,k in enumerate(sub_arms):
        # get /lambda^k
        nu = get_confusing_bandit(k)
        # A_eq[j]=
        for i,idx in enumerate(sub_arms):
            A_ub[j][i] = klBern(thetas[idx], nu[idx])
    A_ub = (-1)*A_ub

    b_ub = np.arange(sub_arms.size, dtype=int)
    b_ub = np.ones_like(b_ub)
    b_ub = (-1)*b_ub

    delta = c[c!=0]

    # option = {'tol':1e-8, 'sym_pos':False, 'cholesky':False, 'lstsq':True}
    # option = {}
    # res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='interior-point', options=option)

    try:
        # option = {'tol':1e-12,'lstsq':True}
        # option={'tol':1e-10, 'sparse':True}
        # option={'tol':1e-10, 'sym_pos':False, 'cholesky':False, 'sparse':True}
        # option = {'tol':tol, 'sym_pos':False, 'cholesky':False, 'lstsq':True}
        option = {'tol':tol}
        res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='interior-point', options=option)
        # option = {}
        # res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='revised simplex', options=option)
    except Exception as e:
        print()
        print("first")
        print(str(e))
        print("delta : ",delta)
        print("A_ub : ", A_ub)
        # print(res)
        return np.full(thetas.size, -1)

    if res.success==True:
        # return res.x
        result = np.zeros(thetas.size)
        for i, idx in enumerate(opt_arms):
            result[idx] = np.inf
        for i, idx in enumerate(sub_arms):
            result[idx] = res.x[i]
        return result
    else:
        print()
        print("second")
        print(res)
        print("thetas : ", thetas)
        print("c : ", c)
        print("delta : ",delta)
        print("A_ub : ", A_ub)
        # print("b_ub : ", b_ub)
        # print("linear programming error!")
        # # return -1
        # return np.full(thetas.size, -1)
        # #sys.exit()
        # we can ignore this failure
        result = np.zeros(thetas.size)
        for i, idx in enumerate(opt_arms):
            result[idx] = np.inf
        for i, idx in enumerate(sub_arms):
            result[idx] = res.x[i]
        return result
##


def solve_optimization_problem__classic(thetas):
    r""" Solve the optimization problem (2)-(3) as defined in the paper, for classical stochastic bandits.

    - No need to solve anything, as they give the solution for classical bandits.
    """
    values = np.zeros_like(thetas)
    theta_max = np.max(thetas)
    for i, theta in enumerate(thetas):
        if theta < theta_max:
            values[i] = 1 / klBern(theta, theta_max)
        else:
            values[i] = np.inf
    return values
    # return 1. / klBern_vect(thetas, np.max(thetas))


##########################################################


class OSSB(object):
    def __init__(self, nbArms, gamma=GAMMA,
                 solve_optimization_problem="classic",
                 lower=0., amplitude=1., **kwargs):

        self.nbArms = nbArms  #: Number of arms

        # Arguments
        assert gamma >= 0, "Error: the 'gamma' parameter for 'OSSB' class has to be >= 0. but was {:.3g}.".format(gamma)  # DEBUG
        self.gamma = gamma  #: Parameter :math:`\gamma` for the OSSB algorithm. Can be = 0.
        
        # Solver for the optimization problem.
        self._solve_optimization_problem = solve_optimization_problem__classic  # Keep the function to use to solve the optimization problem
        self._info_on_solver = ", Bern"  # small delta string
        
        # adding Lipschitz
        if solve_optimization_problem == "Lipschitz":
            self._info_on_solver = ", Lipschitz"
            self._solve_optimization_problem = solve_optimization_problem__Lipschitz
        self._kwargs = kwargs  # Keep in memory the other arguments, to give to self._solve_optimization_problem
        
        # Internal memory
        self.t = 0  #: Internal time
        self.pulls = np.zeros(nbArms, dtype=int)  #: Number of pulls of each arms
        self.rewards = np.zeros(nbArms)  #: Cumulated rewards of each arms
        
        self.phase = None  #: categorical variable for the phase

        self.old_values_c_x_mt = np.full(nbArms, 0)

    def __str__(self):
        """ -> str"""
        return r"OSSB($\gamma={:.3g}${})".format(self.gamma, self._info_on_solver)

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        self.t = 0
        self.pulls.fill(0)
        self.rewards.fill(0)

        self.phase = Phase.initialisation

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
        self.t += 1
        self.pulls[arm] += 1
        self.rewards[arm] += reward

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Applies the OSSB procedure, it's quite complicated so see the original paper."""
        
        # Play each arm once
        if np.any(self.pulls < 1):
            # print("[initial phase] force exploration of an arm that was never pulled...")  # DEBUG
            return np.random.choice(np.nonzero(self.pulls < 1)[0])

        # get estimated mean of each arm
        means = (self.rewards / self.pulls)
        
        # compute the linear programming solution
        # values_c_x_mt = self._solve_optimization_problem(means, **self._kwargs)
        result_mt = self._solve_optimization_problem(means, **self._kwargs)
        if np.all(result_mt == -1):
            print("load previous solution")
            values_c_x_mt = self.old_values_c_x_mt
        else:
            values_c_x_mt = result_mt
            self.old_values_c_x_mt = values_c_x_mt

        # estimation
        underSampledArms = (np.where(self.pulls <= log(self.t)/log(log(self.t))))[0]
        if underSampledArms.size > 0:
            # under-sampled arm
            self.phase = Phase.estimation
            chosen_arm = np.random.choice(underSampledArms)
        else:
            values_c_x_mt[values_c_x_mt > log(self.t)] = log(self.t) # min{\eta_{n,i}, log(n)}
            values_c_x_mt = (1+self.gamma) * values_c_x_mt * log(self.t)
            values = values_c_x_mt - self.pulls
            # exploitation
            if np.all(values<=0):
                self.phase = Phase.exploitation
                # current best arm
                chosen_arm = np.random.choice(np.nonzero(means == np.max(means))[0])
            # exploration
            else:
                self.phase = Phase.exploration
                max_value = np.max(values)
                # most under-explored arm
                chosen_arm = np.random.choice(np.nonzero(values==max_value)[0])

        return chosen_arm


class LipschitzOSSB(OSSB):
    def __init__(self, nbArms, gamma=GAMMA, L=-1, lower=0, amplitude=1., **kwargs):
        # if L==-1:
        #     L = estimate_Lipschitz_constant()
        kwargs.update({'L': L})
        super(LipschitzOSSB, self).__init__(nbArms, gamma=gamma, solve_optimization_problem="Lipschitz", lower=lower, amplitude=amplitude, **kwargs)