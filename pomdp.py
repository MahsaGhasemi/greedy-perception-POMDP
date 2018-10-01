import time
import numpy as np
import random

def sample_disc_dist(dist):
    """Samples a point from a discrete distribution"""

    # dist -- an array of probability mass

    sample = np.random.uniform()
    dist_sum = 0.0
    for i, m in enumerate(dist):
        dist_sum += m
        if sample < dist_sum:
            return i
    return i

def cart_prod(sets):
    """Generate n-fold Cartesian product of sets"""

    set_prod = [(elem,) for elem in sets[0]]
    for s in sets[1:]:
        # find product of a new set with previous product set
        set_prod = [elem + (new_elem,) for elem in set_prod for new_elem in s]

    return set_prod

class Pomdp:
    """Creates a POMDP object"""

    def __init__(self, states=None, actions=None, transitions=None,
                 reward=None, discount=None,
                 observations=None, observation_func=None, sel_card=None):
        self.states = states # states
        self.actions = actions  # actions
        # transition function, array of size (#state,#action,#state)
        # with probability value
        self.transitions = transitions
        # reward function, array of size (#state,#action) with reward value
        self.reward = reward
        self.discount = discount # discount factor
        self.observations = observations # observations
        # observation function, array of size (#state,#action,#observation)
        # with probability value
        self.observation_func = observation_func
        self.sel_card = sel_card

    class Policy:
        """Creates a policy for a POMDP as a piecewise-linear and convex value
        function; a 3-tuple (belief_points, gradients, optimal_actions)"""

        def __init__(self, belief_set, gr_vectors, opt_actions, perc_method, sel_card):
            self.belief_set = belief_set
            self.gr_vectors = gr_vectors
            self.opt_actions = opt_actions
            self.perc_method = perc_method
            self.sel_card = sel_card

        def online_perception(self,belief,action,aux_observation_func):
            """Decides about perception action"""
            if self.perc_method == 'greedy':
                action_sel = self.greedy_opt(belief,action,aux_observation_func)
            elif self.perc_method == 'random':
                action_sel = self.random_sel(belief,action,aux_observation_func)

            return action_sel

        def compute_gain(self, belief, action, action_sel, aux_observation_func, obs):
            """Computes marginal gain of adding an observation"""

            gain = 0.0
            act_ext = action_sel+[obs]

            n_state = int(aux_observation_func.shape[1])
            observations = np.arange(int(aux_observation_func.shape[-1]))

            for o_val in cart_prod([observations for c in action_sel+[obs]]):
                temp_val = np.empty(n_state)
                for i_s in range(n_state):
                    temp_val[i_s] = belief[i_s]*np.prod(
                            [aux_observation_func[act_ext[io_sel],i_s,action,o_sel]
                            for io_sel, o_sel in enumerate(o_val)])

                temp_sum = np.sum(temp_val)
                gain += np.sum([temp_val[t]*np.log(temp_val[t]/temp_sum)
                                for t in np.nonzero(temp_val)])

            return -gain

        def greedy_opt(self, belief, action, aux_observation_func):
            """Selects a near-optimal subset of observations for
            belief entropy minimization"""

            rem_obs = range(len(aux_observation_func)) # remaining set of observations
            action_sel = [] # set of selected observations
            for j in range(self.sel_card):
                marginal_gains = [self.compute_gain(belief, action, action_sel,
                                                    aux_observation_func, obs)
                                  for obs in rem_obs]
                ind_to_add = np.argmin(marginal_gains)
                action_sel.append(rem_obs[ind_to_add])
                rem_obs.pop(ind_to_add)

            return np.array(action_sel)

        def random_sel(self, belief, action, aux_observation_func):
            """Selects a random subset of observations"""

            action_sel = random.sample(range(len(aux_observation_func)),
                                       self.sel_card)

            return np.array(action_sel)

    def simulate(self, policy, init_belief, init_state, horizon, goal=None):
        """Simulates a run in POMDP"""

        # policy -- a 4-tuple (belief_points, gradients,
        #                      optimal_actions, selection_actions)
        # init_belief -- an array with size=#states
        # init_state -- index of initial state of the agent

        state = np.empty(horizon+1, dtype=int) # true state
        belief_temp = np.empty((horizon+1,len(self.states)))
        belief = np.empty((horizon+1,len(self.states))) # agent's belief
        action = np.empty(horizon, dtype=int) # planning actions
        action_sel = np.empty((horizon,self.sel_card), dtype=int) # perception actions
        observation = np.empty(horizon, dtype=int) # observations
        aux_observation = np.empty((horizon,self.sel_card), dtype=int) # auxiliary observations
        reward = np.empty(horizon+1, dtype=float) # reward

        t = 0 # time step
        state[t] = init_state
        belief_temp[t] = init_belief
        belief[t] = init_belief
        t_sel_avg = 0.0

        for t in range(horizon):
            print "time step: "+str(t+1)+'\n'
            opt_vec = np.argmax([np.dot(gr_vec, belief[t]) for gr_vec in policy.gr_vectors])
            action[t] = policy.opt_actions[opt_vec]
            reward[t] = np.power(self.discount,t) * self.reward[state[t],action[t]]
            state[t+1] = sample_disc_dist(self.transitions[state[t],action[t]])
            observation[t] = sample_disc_dist(self.observation_func[
                                              state[t+1],action[t]])
            belief_temp[t+1] = self.update_belief(belief[t],action[t],
                                                  observation[t])
            aux_observation_func = aux_agents(t)
            t_s = time.time()
            action_sel[t] = policy.online_perception(belief_temp[t+1],action[t],
                                                     aux_observation_func)
            t_f = time.time()
            t_sel_avg += (t_f-t_s)
            aux_observation[t] = [sample_disc_dist(aux_observation_func[
                    o_sel,state[t+1],action[t]]) for o_sel in action_sel[t]]
            belief[t+1] = self.update_belief_obs(belief_temp[t+1],action[t],
                                            action_sel[t],aux_observation_func,
                                            aux_observation[t])

            # terminate when the goal is reached
            if goal != None and state[t+1]==goal:
                t_end = t + 1
                break
            else:
                t_end = t + 1

        reward[t_end] = np.power(self.discount,t_end) * self.reward[state[t_end],4] # if stops at last state
        t_sel_avg /= t_end
        path = (state, belief, action_sel, action, observation, reward,
                t_end, t_sel_avg)

        return path

    def update_belief(self, belief, action, observation):
        """Updates the belief based on current belief,
        action, and observation"""

        belief = np.multiply(self.observation_func[:,action,observation],
                np.dot(np.transpose(self.transitions[:,action,:]), belief))
        belief = belief/np.linalg.norm(belief, ord=1)

        return belief

    def update_belief_obs(self, belief, action, action_sel,
                          aux_observation_func, observation):
        """Updates the belief based on current belief,
        perception action, and observation"""

        belief = np.multiply(np.prod(
                 [aux_observation_func[o_sel,:,action,observation[io_sel]]
                 for io_sel, o_sel in enumerate(action_sel)], axis=0),
                 belief)
        belief = belief/np.linalg.norm(belief, ord=1)

        return belief