"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False): #if True, your class is allowed to print debugging statements
        self.verbose = verbose ###
        self.num_actions = num_actions ###
        self.s = 0 ### state
        self.a = 0 ### action
        self.num_states=num_states
        self.num_actions=num_actions
        self.alpha=alpha
        self.gamma = gamma
        self.rar=rar
        self.radr=radr
        self.dyna=dyna
        self.Qmat=np.zeros((self.num_states,self.num_actions))
        self.QmatOld=np.ones((self.num_states,self.num_actions))
        self.data={}


    def querysetstate(self, s):
        #sets the state to s, and returns an integer
        #action according to the same rules as query()

        #does not execute an update to the Q-table
        #does not update rar

        #1) To set the initial state, and
        #2) when using a learned policy, but not updating it.
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if self.rar > rand.uniform(0,1):
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Qmat[self.s,:])
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        #Keep track of the last state s and the last action a,
        #then use the new information s_prime and r to update the Q table

        # learning instance, or experience tuple is <s, a, s_prime, r>.
        #query() should return an integer, which is the next action to take

        #Choose a random action with probability rar, and that it should
        #update rar according to the decay rate radr at each step

        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """
        for i in range(self.dyna):
            #self.data[self.s,self.a].append(s_prime,r)
            if (self.s,self.a) in self.data:
                self.data[(self.s,self.a)].append((s_prime,r))
            else:
                self.data[(self.s,self.a)]=[]
                self.data[(self.s,self.a)].append((s_prime,r))
            #self.data.setdefault([self.s][self.a],[]).append({'s_prime':'r'})
            row_indx=rand.randint(0,len(self.data)-1)
            value_at_index = self.data.values()[row_indx]
            key_at_index=self.data.keys()[row_indx]
            col_indx=rand.randint(0,len(value_at_index)-1)
            sprime_fake=value_at_index[col_indx][0]
            r_fake=value_at_index[col_indx][1]
            s_fake=key_at_index[0]
            a_fake=key_at_index[1]
            IndexOfArgmaxA_fake=np.argmax(self.Qmat[sprime_fake,:])
            LearnedValue_fake=r_fake+ self.gamma*self.Qmat[sprime_fake,IndexOfArgmaxA_fake]
            self.Qmat[s_fake,a_fake]=(1-self.alpha)*self.Qmat[s_fake,a_fake]+self.alpha*LearnedValue_fake

#        MATRIC METHOD
#        if self.dyna:
#            self.data.setdefault((self.s,self.a),[]).append(s_prime,r)
#            sprime_fake = np.random.choice(range(len(self.data.keys())), self.dyna)
#            sprime_fake = np.asarray(self.data.keys())[sprime_fake]
#            rew_fake = np.asarray([rand.choice(self.data[tuple(x)]) for x in sprime_fake])
#            Qtemp_fake = self.Qmat[rew_fake[:, 0], np.argmax(self.Qmat[rew_fake[:, 0], :], axis = 1)]
#            self.Qmat[sprime_fake[:,0], sprime_fake[:,1]] = (1 - self.alpha) * self.Qmat[sprime_fake[:,0], sprime_fake[:,1]] + self.alpha * ((rew_fake)[:,1] + self.gamma * Qtemp_fake)


        prev_state=self.s
        new_state=s_prime
        reward=r
        self.rar=self.rar*self.radr
        IndexOfArgmaxA=np.argmax(self.Qmat[new_state,:])
        #LearnedValue=reward+self.gamma*self.Qmat[new_state,IndexOfArgmaxA] ##Qlearning
        LearnedValue=reward+ self.gamma*self.Qmat[new_state,IndexOfArgmaxA] ##DynaQlearning
        ##Update QMatrix based on tuple(old action)
        self.QmatOld=self.Qmat ## store the old Q matrix for terminal condition
        self.Qmat[prev_state,self.a]=(1-self.alpha)*self.Qmat[prev_state,self.a]+self.alpha*LearnedValue
        self.s=new_state ##Update the state
        #action = querysetstate(self.s)
        #self.a=action
        #if np.absolute(np.linalg.det(self.Qmat-self.QmatOld))<=0.01:
        #    break
        if self.rar > rand.uniform(0,1):
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Qmat[self.s,:])
        self.a= action

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action


    def author(self):
        return 'apatel380'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
