from OSSB import OSSB, LipschitzOSSB, estimate_Lipschitz_constant
from itertools import combinations
import numpy as np
from numpy.random import binomial
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os

case=2
if case==1:
    arms = np.array([0.8, 0.6, 0.21, 0.22, 0.23, 0.24])
else:
    arms = np.array([0.5,0.4,0.4,0.4,0.51,0.4,0.4,0.4])

horizon = 10000
repetitions = 50
# delta_t_plot =    # sampling rate for plotting
nbPolicies = 3
useJoblib = True
n_jobs = 10


GAMMA = 0.001

# Evaluator
# Internal vectorial memory
allPulls = np.zeros((nbPolicies, arms.size, horizon), dtype=np.int32)
arms_pulls = np.zeros((nbPolicies, repetitions, arms.size))



class Result(object):
    """ Result accumulators."""
    def __init__(self, nbArms, horizon):
        self.choices = np.zeros(horizon, dtype=int)  #: Store all the choices.
        self.rewards = np.zeros(horizon)  #: Store all the rewards, to compute the mean.
        self.pulls = np.zeros(nbArms, dtype=int)  #: Store the pulls.

    def store(self, time, choice, reward):
        """ Store results."""
        self.choices[time] = choice
        self.rewards[time] = reward
        self.pulls[choice] += 1


def compute_cache_rewards(arms):
    """ Compute only once the rewards, then launch the experiments with the same matrix (r_{k,t})."""
    rewards = np.zeros((len(arms), repetitions, horizon))
    print("\n===> Pre-computing the rewards ... Of shape {} ...\n    In order for all simulated algorithms to face the same random rewards (robust comparison of A1,..,An vs Aggr(A1,..,An)) ...\n".format(np.shape(rewards)))  # DEBUG
    for armId, arm in enumerate(arms):
        rewards[armId] = np.asarray(binomial(1, arm, (repetitions,horizon)), dtype=float)
    return rewards


def delayed_play(policy, horizon,
                    random_shuffle, random_invert, nb_break_points,
                    seed=None, allrewards=None, repeatId=0,
                    useJoblib=False):\
    # Start game
    policy.startGame()
    result = Result(arms.size, horizon)  # One Result object, for every policy
    prettyRange = tqdm(range(horizon), desc="Time t") if repeatId == 0 else range(horizon)
    for t in prettyRange:
        # 1. The player's policy choose an arm
        choice = policy.choice()
        # 2. A random reward is drawn, from this arm at this time
        reward = allrewards[choice, repeatId, t]
        # 3. The policy sees the reward
        policy.getReward(choice, reward)
        # 4. Finally we store the results
        result.store(t, choice, reward)
    return result

def store(r, policyId, repeatId):
    """ Store the result of the #repeatId experiment, for the #policyId policy."""
    allPulls[policyId, :, :] += np.array([1 * (r.choices == armId) for armId in range(arms.size)])  # XXX consumes a lot of zeros but it is not so costly
    arms_pulls[policyId,repeatId,:] = r.pulls


def get_allMeans(horizon):
    allMeans = np.zeros((arms.size, horizon))
    for t in range(horizon):
        allMeans[:,t]=arms
    return allMeans

def getAverageWeightedSelections(policyId):
    weighted_selections = np.zeros(horizon)
    for armId in range(arms.size):
        mean_selections = allPulls[policyId, armId, :] / float(repetitions)
        meanOfThisArm = get_allMeans(horizon)[armId, :]
        weighted_selections += meanOfThisArm * mean_selections
    return weighted_selections

def getCumulatedRegret(policyId):
    instant_oracle_performance = np.full(horizon, np.amax(arms))
    instant_performance = getAverageWeightedSelections(policyId)
    instant_loss = instant_oracle_performance - instant_performance
    return np.cumsum(instant_loss)
_times = np.arange(1, 1 + horizon)
def plotRegrets():
    X = _times - 1

    labels = ["Lipschitz_known", "Lipschitz_unknown", "OSSB_no_structure"]
    plt.figure()
    for policyId in range(nbPolicies):
        Y = getCumulatedRegret(policyId)
        plt.plot(X, Y, label = labels[policyId])
    plt.xlabel('time')
    plt.ylabel('cumulated regret')
    plt.title("cumulated regret with GAMMA:{}, repetitions:{}".format(GAMMA, repetitions))
    plt.legend()
    plt.savefig(dir_name+"/cumulated regret with GAMMA:{}, repetitions:{}.png".format(GAMMA, repetitions), dpi=300)
    # plt.show()
    plt.close()

    # plt.figure()
    # for policyId in range(2):
    #     Y = getCumulatedRegret(policyId)
    #     plt.plot(X, Y, label = labels[policyId])
    # plt.xlabel('time')
    # plt.ylabel('cumulated regret')
    # plt.title("cumulated regret with EPSILON:{}, GAMMA:{}, repetitions:{}".format(EPSILON, GAMMA, repetitions))
    # plt.legend()
    # plt.savefig(dir_name+"/cumulated regret 2 with EPSILON:{}, GAMMA:{}, repetitions:{}.png".format(EPSILON, GAMMA, repetitions), dpi=300)
    # # plt.show()
    # plt.close()

    # plot the number of each arms
    X = np.arange(arms.size)
    plt.figure()
    for policyId in range(nbPolicies):
        minPull = arms_pulls[policyId].min(axis=0)
        # plt.plot(X, Ymin, label = labels[policyId])

        maxPull = arms_pulls[policyId].max(axis=0)
        # plt.plot(X, Ymax, label = labels[policyId])

        meanPull = arms_pulls[policyId].mean(axis=0)
        plt.plot(X, meanPull, label = labels[policyId])
        plt.fill_between(X, minPull, maxPull, alpha=0.3)
    plt.xlabel('arms')
    plt.ylabel('number of pulls')
    plt.title('number of pulls with GAMMA:{}, repetitions:{}'.format(GAMMA, repetitions))
    plt.legend()
    plt.savefig(dir_name+'/number of pulls', dpi=300)
    plt.close()

if __name__ == '__main__':
    dir_name = "{},{},case{}".format(GAMMA, repetitions, case)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # pre calculate the arms' reward
    allrewards = compute_cache_rewards(arms)
    # print(allRewards)

    policies = []

    # LipschitzOSSB with known true Lipschitz constant
    L = estimate_Lipschitz_constant(arms)   # true Lipschitz Constant
    trueLC = L
    print(trueLC)
    policies.append(LipschitzOSSB(arms.size, GAMMA, L, 0, 1.))

    # LipschitzOSSB with unknown Lipschitz constant, estimating it during process
    policies.append(LipschitzOSSB(arms.size, GAMMA, -1, 0, 1.))

    # classic OSSB without any structure
    policies.append(OSSB(arms.size, GAMMA, lower=0, amplitude=1.))


    for policyId, policy in enumerate(policies):
        print(policyId, policy)
        if useJoblib:
            seeds = np.random.randint(low=0, high=100 * repetitions, size=repetitions)
            repeatIdout = 0
            for r in Parallel(n_jobs=n_jobs, pre_dispatch='3*n_jobs', verbose=6)(
                delayed(delayed_play)(policy, horizon, random_shuffle=False, random_invert=False, nb_break_points=0, allrewards=allrewards, seed=seeds[repeatId], repeatId=repeatId, useJoblib=useJoblib)
                for repeatId in tqdm(range(repetitions), desc="Repeat||")
            ):
                store(r, policyId, repeatIdout)
                repeatIdout += 1
        else:
            for repeatId in tqdm(range(repetitions), desc="Repeat"):
                r = delayed_play(policy, horizon, random_shuffle=False, random_invert=False, nb_break_points=0, allrewards=allrewards, repeatId=repeatId, useJoblib=useJoblib)
                store(r, policyId, repeatId)
    
    plotRegrets()