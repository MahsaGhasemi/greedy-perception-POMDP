import numpy as np

class Pbvi:
    """A point-based value iteration solver"""

    def __init__(self, pomdp):
        self.pomdp = pomdp

    def init_belief_set(self, method='det-uniform', fold=0, n_samples=10):
        """Picks the initial set of belief points"""

        if method == 'det-uniform':
            self.belief_set = np.eye(len(self.pomdp.states))
            for f in range(fold):
                self.belief_set = np.unique(
                        np.concatenate((self.belief_set,
                        np.mean(
                        [(self.belief_set[i],self.belief_set[j])
                         for i in range(len(self.belief_set))
                         for j in range(i)],
                        axis=1)), axis=0), axis=0)

        elif method == 'rand-uniform':
            self.belief_set = np.diff(np.concatenate(
                    (np.zeros((n_samples,1)),
                     np.sort(np.random.uniform(
                             0,1,(n_samples,len(self.pomdp.states)-1))),
                     np.ones((n_samples,1))), axis=1), axis=1)

        # set initial value function to lowest possible value
        v0 = 1/(1-self.pomdp.discount)*np.min(self.pomdp.reward,axis=0)[-1] # minimum reward value
        # value function at belief points;
        # initially set to lowest possible value
        self.value = v0*np.ones(len(self.belief_set))
        # gradient vectors of the value function at belief points;
        # initially set by the constant lowest possible value
        self.gr_vectors = np.tile(v0*np.ones(len(self.pomdp.states)),
                                  (len(self.belief_set),1))
        # optimal actions at belief points;
        # initially set to first action
        self.opt_actions = np.zeros(len(self.belief_set), dtype=int)

    def bellman_backup(self):
        """Applies a Bellman operator backup on the current belief set"""

        tau_a_b = np.empty([len(self.pomdp.actions),
                            len(self.belief_set),
                            len(self.pomdp.states)])
        for ia, a in enumerate(self.pomdp.actions):
            # tau with size n_o*n_b*n_s
            tau_a_o = np.array([[self.pomdp.discount *
                                 np.dot(self.pomdp.transitions[:,ia,:] *
                                        self.pomdp.observation_func[:,ia,io],
                                        gr_vec)
                                 for gr_vec in self.gr_vectors]
                                for io in range(len(self.pomdp.observations))])

            for ib, b in enumerate(self.belief_set):
                # tau with size n_a*n_b*n_s
                tau_a_b[ia,ib] = self.pomdp.reward[:,ia] +\
                                 np.sum([tau_a_o[io, np.argmax([np.dot(tau, b)
                                                 for tau in tau_a_o[io]])]
                                         for io in range(len(self.pomdp.observations))],
                                        axis=0)

        for ib, b in enumerate(self.belief_set):
            self.opt_actions[ib] = np.argmax([
                    np.dot(tau_a_b[ia,ib], b)
                    for ia, a in enumerate(self.pomdp.actions)])
            self.gr_vectors[ib] = tau_a_b[self.opt_actions[ib], ib]
            self.value[ib] = np.dot(self.gr_vectors[ib],b)