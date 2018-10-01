import time
import numpy as np
from pomdp import *
from pbvi_solver import *

def robot(n_state=8, goal=64, discount=0.95, sel_card=1):
    """Generates a POMDP for a robot moving in a 2-D map"""

    if isinstance(n_state, int):
        n1 = n_state; n2 = n_state
    else:
        n1, n2 = n_state

    states = np.arange(n1*n2)
    obstacles = [4,10,14,21,25,35,39,52,57,62]
    actions = np.arange(5) # 0 is up, 1 is right, 2 is down, 3 is left, and 4 is stop
    transitions = np.zeros((len(states),len(actions),len(states)))
    for s in states:
        for a in actions:

            if a == 0:
                if s in [56,57,58,59,60,61,62,63]:
                    transitions[s,a,s] = 1
                elif s == 0:
                    transitions[s,a,8] = 0.7
                    transitions[s,a,1] = 0.3
                elif s == 7:
                    transitions[s,a,15] = 0.7
                    transitions[s,a,6] = 0.3
                elif s in [1,2,3,4,5,6]:
                    transitions[s,a,s+8] = 0.7
                    transitions[s,a,s-1] = 0.15
                    transitions[s,a,s+1] = 0.15
                elif s in [15,23,31,39,47,55]:
                    transitions[s,a,s+8] = 0.7
                    transitions[s,a,s-8] = 0.15
                    transitions[s,a,s-1] = 0.15
                elif s in [8,16,24,32,40,48]:
                    transitions[s,a,s+8] = 0.7
                    transitions[s,a,s-8] = 0.15
                    transitions[s,a,s+1] = 0.15
                else:
                    transitions[s,a,s+8] = 0.7
                    transitions[s,a,s-8] = 0.1
                    transitions[s,a,s-1] = 0.1
                    transitions[s,a,s+1] = 0.1

            elif a == 1:
                if s in [7,15,23,31,39,47,55,63]:
                    transitions[s,a,s] = 1
                elif s == 0:
                    transitions[s,a,1] = 0.7
                    transitions[s,a,8] = 0.3
                elif s == 56:
                    transitions[s,a,57] = 0.7
                    transitions[s,a,48] = 0.3
                elif s in [8,16,24,32,40,48]:
                    transitions[s,a,s+1] = 0.7
                    transitions[s,a,s-8] = 0.15
                    transitions[s,a,s+8] = 0.15
                elif s in [57,58,59,60,61,62]:
                    transitions[s,a,s+1] = 0.7
                    transitions[s,a,s-8] = 0.15
                    transitions[s,a,s-1] = 0.15
                elif s in [1,2,3,4,5,6]:
                    transitions[s,a,s+1] = 0.7
                    transitions[s,a,s-1] = 0.15
                    transitions[s,a,s+8] = 0.15
                else:
                    transitions[s,a,s+1] = 0.7
                    transitions[s,a,s-8] = 0.1
                    transitions[s,a,s-1] = 0.1
                    transitions[s,a,s+8] = 0.1

            elif a == 2:
                if s in [0,1,2,3,4,5,6,7]:
                    transitions[s,a,s] = 1
                elif s == 56:
                    transitions[s,a,48] = 0.7
                    transitions[s,a,57] = 0.3
                elif s == 63:
                    transitions[s,a,55] = 0.7
                    transitions[s,a,62] = 0.3
                elif s in [57,58,59,60,61,62]:
                    transitions[s,a,s-8] = 0.7
                    transitions[s,a,s-1] = 0.15
                    transitions[s,a,s+1] = 0.15
                elif s in [15,23,31,39,47,55]:
                    transitions[s,a,s-8] = 0.7
                    transitions[s,a,s-1] = 0.15
                    transitions[s,a,s+8] = 0.15
                elif s in [8,16,24,32,40,48]:
                    transitions[s,a,s-8] = 0.7
                    transitions[s,a,s+1] = 0.15
                    transitions[s,a,s+8] = 0.15
                else:
                    transitions[s,a,s-8] = 0.7
                    transitions[s,a,s+1] = 0.1
                    transitions[s,a,s-1] = 0.1
                    transitions[s,a,s+8] = 0.1

            elif a == 3:
                if s in [0,8,16,24,32,40,48,56]:
                    transitions[s,a,s] = 1
                elif s == 7:
                    transitions[s,a,6] = 0.7
                    transitions[s,a,15] = 0.3
                elif s == 63:
                    transitions[s,a,62] = 0.7
                    transitions[s,a,55] = 0.3
                elif s in [15,23,31,39,47,55]:
                    transitions[s,a,s-1] = 0.7
                    transitions[s,a,s-8] = 0.15
                    transitions[s,a,s+8] = 0.15
                elif s in [1,2,3,4,5,6]:
                    transitions[s,a,s-1] = 0.7
                    transitions[s,a,s+1] = 0.15
                    transitions[s,a,s+8] = 0.15
                elif s in [57,58,59,60,61,62]:
                    transitions[s,a,s-1] = 0.7
                    transitions[s,a,s+1] = 0.15
                    transitions[s,a,s-8] = 0.15
                else:
                    transitions[s,a,s-1] = 0.7
                    transitions[s,a,s+1] = 0.1
                    transitions[s,a,s-8] = 0.1
                    transitions[s,a,s+8] = 0.1

            else: # a == 4
                transitions[s,a,s] = 1

    reward = np.zeros((len(states),len(actions))) - 1 # discourage wandering
    reward[obstacles] = -5 # reward at obstacles
    reward[goal-1] = 10 # reward at terminal state

    observations = np.arange(len(states))
    observation_func = np.empty((len(states),len(actions),len(observations)))
    temp_a = 0
    p_correct = 0.5; p_out = 0.1
    for s in states:
        if s in [0,7,56,63]:
                if s == 0:
                    cover = [1,8,9]
                elif s == 7:
                    cover = [6,14,15]
                elif s == 56:
                    cover = [48,49,57]
                else: # s == 63
                    cover = [54,55,62]
        elif s in [1,2,3,4,5,6]:
            cover = [s-1,s+1,s+7,s+8,s+9]
        elif s in [15,23,31,39,47,55]:
            cover = [s-9,s-8,s-1,s+7,s+8]
        elif s in [57,58,59,60,61,62]:
            cover = [s-9,s-8,s-7,s-1,s+1]
        elif s in [8,16,24,32,40,48]:
            cover = [s-8,s-7,s+1,s+8,s+9]
        else:
            cover = [s-9,s-8,s-7,s-1,s+1,s+7,s+8,s+9]

        for o in observations:
            if s == o:
                observation_func[s,temp_a,o] = p_correct
            elif o in cover:
                observation_func[s,temp_a,o] = (1-p_correct-p_out)/len(cover)
            else:
                observation_func[s,temp_a,o] = p_out/(len(states)-1-len(cover))

        # indifference of observations to actions
        for ia in range(1,len(actions)):
            observation_func[:,ia,:] = observation_func[:,temp_a,:]

    return Pomdp(states, actions, transitions, reward, discount,
                 observations, observation_func, sel_card)

def aux_agents(ts):
    """Simulates auxilliary agents and their observations"""

    n_state = 64
    path1 = [1 ,2 ,3 ,4 ,5 ,6 ,7 ,
             8 ,16,24,32,40,48,56,
             64,63,62,61,60,59,58,
             57,49,41,33,25,17,9 ]
    lp1 = 28
    path2 = [10,11,12,13,14,
             15,23,31,39,47,
             55,54,53,52,51,
             50,42,34,26,18]
    lp2 = 20
    path3 = [19,20,21,
             22,30,38,
             46,45,44,
             43,35,27]
    lp3 = 12

    n_agent = 12 # number of agents
    pos_agents = np.empty(n_agent, dtype=int)
    # agents 0:3 --> path1; agents 4-7 --> path2; agents 8-11 --> path3
    phase = [0 ,7 ,14,21,
             0 ,5 ,10,15,
             0 ,3 ,6 , 9]
    n_observation = n_state
    cover_all = np.empty((n_agent,9), dtype=int)
    observations = np.arange(1,n_observation+1)
    n_action = 5
    aux_observation_func = np.empty((n_agent,n_state,n_action,n_observation))
    temp_a = 0

    for i in range(4):
        mode = (ts+phase[i]) % lp1
        pos_agents[i] = path1[mode]
        if pos_agents[i] in [1,8,57,64]:
            if pos_agents[i] == 1:
                cover = [1,2,3,9,10,11,17,18,19]
                cover_all[i] = cover
            elif pos_agents[i] == 8:
                cover = [6,7,8,14,15,16,22,23,24]
                cover_all[i] = cover
            elif pos_agents[i] == 57:
                cover = [41,42,43,49,50,51,57,58,59]
                cover_all[i] = cover
            else: # pos_agents[i] == 64:
                cover = [46,47,48,54,55,56,62,63,64]
                cover_all[i] = cover
        else:
            if pos_agents[i] in [2,3,4,5,6,7]:
                cover = [pos_agents[i]-1,pos_agents[i],pos_agents[i]+1,
                         pos_agents[i]+7,pos_agents[i]+8,pos_agents[i]+9,
                         pos_agents[i]+15,pos_agents[i]+16,pos_agents[i]+17]
                cover_all[i] = cover
            if pos_agents[i] in [16,24,32,40,48,56]:
                cover = [pos_agents[i]-10,pos_agents[i]-9,pos_agents[i]-8,
                         pos_agents[i]-2,pos_agents[i]-1,pos_agents[i],
                         pos_agents[i]+6,pos_agents[i]+7,pos_agents[i]+8]
                cover_all[i] = cover
            if pos_agents[i] in [58,59,60,61,62,63]:
                cover = [pos_agents[i]-17,pos_agents[i]-16,pos_agents[i]-15,
                         pos_agents[i]-9,pos_agents[i]-8,pos_agents[i]-7,
                         pos_agents[i]-1,pos_agents[i],pos_agents[i]+1]
                cover_all[i] = cover
            else: # pos_agents[i] in [9,17,25,33,41,49]
                cover = [pos_agents[i]-8,pos_agents[i]-7,pos_agents[i]-6,
                         pos_agents[i],pos_agents[i]+1,pos_agents[i]+2,
                         pos_agents[i]+8,pos_agents[i]+9,pos_agents[i]+10]
                cover_all[i] = cover

    for i in range(4,8):
        mode = (ts+phase[i]) % lp2
        pos_agents[i] = path2[mode]
        cover = [pos_agents[i]-9,pos_agents[i]-8,pos_agents[i]-7,
                 pos_agents[i]-1,pos_agents[i],pos_agents[i]+1,
                 pos_agents[i]+7,pos_agents[i]+8,pos_agents[i]+9]
        cover_all[i] = cover

    for i in range(8,12):
        mode = (ts+phase[i]) % lp3
        pos_agents[i] = path3[mode]
        cover = [pos_agents[i]-9,pos_agents[i]-8,pos_agents[i]-7,
                 pos_agents[i]-1,pos_agents[i],pos_agents[i]+1,
                 pos_agents[i]+7,pos_agents[i]+8,pos_agents[i]+9]
        cover_all[i] = cover

    p_correct = 0.75; p_out = 0.05
    for i in range(n_agent):
        for ss in observations:
            if ss in cover_all[i]:
                for o in observations:
                    if ss == o:
                        aux_observation_func[i,ss-1,temp_a,o-1] = p_correct
                    elif o in cover_all[i]:
                        aux_observation_func[i,ss-1,temp_a,o-1] = (1-p_correct-p_out)/8
                    else:
                        aux_observation_func[i,ss-1,temp_a,o-1] = p_out/(n_state-9)
            else:
                for o in observations:
                    aux_observation_func[i,ss-1,temp_a,o-1] = 1.0/n_state

    # indifference of observations to actions
    for ia in range(1,n_action):
        aux_observation_func[:,:,ia,:] = aux_observation_func[:,:,temp_a,:]

    return aux_observation_func

if __name__ == '__main__':

    # define POMDP
    pomdp = robot()

    # define algorithm parameters
    term_flag = False # termination condition satisfied or not
    it = 0 # iteration counter
    max_it = 100 # maximum number of iterations
    val_th = 0.001 # normalized-value threshold for termination
    norm_type = 1 # type of norm to apply for termination condition

    # initialize solver
    solver = Pbvi(pomdp)
    solver.init_belief_set()

    val_old = np.copy(solver.value)

    t_start = time.time()
    while not term_flag:
        # backup
        solver.bellman_backup()
        val_new = np.copy(solver.value)

        # check termination conditions
        it += 1; print "iteration: "+str(it)+'\n'
        if np.linalg.norm(val_old-val_new, ord=norm_type)/\
           np.linalg.norm(val_old, ord=norm_type) < val_th :
            term_flag = True

        elif it >= max_it:
            term_flag = True

        val_old = np.copy(val_new)

    t_end = time.time()
    t_solver = t_end-t_start
    print "running time: "+str(t_solver)+'\n'

    # extract policy
    belief_set = solver.belief_set
    gr_vectors = solver.gr_vectors
    opt_actions = solver.opt_actions
    policy = pomdp.Policy(belief_set,gr_vectors,opt_actions,'random',pomdp.sel_card)

    # simulate
    n_sim = 50 # number of simulation runs
    avg_reward = 0.0 # average reward over simulation runs
    init_belief = 1.0/len(pomdp.states)*np.ones(len(pomdp.states))
    init_state = 0
    horizon = 40
    hist_result = []
    t_move = 0
    rew_hist = []
    ent_hist = []
    for it in range(n_sim):
        print "simulation: "+str(it+1)+'\n'
        sim_result = pomdp.simulate(policy,init_belief,init_state,horizon,goal=63)
        hist_result.append(sim_result)
        t_end = sim_result[6]
        t_move += t_end
        bel = sim_result[1][:t_end+1]
        ent_hist.append(-np.sum(np.multiply(bel,np.log(bel)),axis=1))
        reward = np.sum(sim_result[5][:t_end+1])
        rew_hist.append(reward)
        avg_reward += reward
    avg_reward /= n_sim
    t_move /= n_sim
    std = np.std(rew_hist)
    print "moving time: "+str(t_move)+'\n'
    print "average reward: "+str(avg_reward)+'\n'
    print "standard deviation of reward: "+str(std)+'\n'

    ent_mean = np.zeros(horizon)
    ent_std = np.zeros(horizon)
    for t in range(horizon):
        num = 0
        temp_ent = []
        for h in ent_hist:
            if len(h)>t:
                num += 1
                ent_mean[t] += h[t]
                temp_ent.append(h[t])
            else:
                pass
        if num == 0:
            ent_mean[t] = None
        else:
            ent_mean[t] /= num
            ent_std[t] = np.std(temp_ent)
    print "average entropy: "+str(ent_mean)+'\n'
    print "standard deviation of entropy: "+str(ent_std)+'\n'