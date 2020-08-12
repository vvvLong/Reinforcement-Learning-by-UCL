#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:24:36 2019

@author: victor
"""

#%% import libs

import numpy as np
import pandas as pd
 
from Environment import Easy21

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#%% Monte-Carlo Control

class MC_Control(object):
    """
    
    Initialise the value function to zero. Use a time-varying scalar step-size 
    of αt = 1/N(st,at) and an ε-greedy exploration strategy with 
    εt = N0/(N0 + N(st)), where N0 = 100 is a constant, N(s) is the number of 
    times that state s has been visited, and N(s,a) is the number of times 
    that action a has been selected from state s. Feel free to choose an 
    alternative value for N0, if it helps producing better results. Plot the 
    optimal value function V ∗ (s) = maxa Q∗ (s, a) using similar axes to the 
    following figure taken from Sutton and Barto’s Blackjack example.
    
    """
    
    def __init__(self, N0=100):
         self._env = Easy21()
         self._actions = self._env.actionSpace()
         self._N0 = N0
         self._Q = np.zeros((11,22,2)) # action-value function, tabular
         self._Nsa = np.zeros((11,22,2)) # number of times s,a has been selected
         
         # number of times s has been visited
         self._Ns = lambda d,p: sum(self._Nsa[d,p])
         
         # ε of each state
         self._eps = lambda d,p: self._N0/(self._N0 + self._Ns(d,p))
         
         # alpha of each s,a pair
         self._alpha = lambda d,p,a: 1/self._Nsa[d,p,a]
         
    def _epsGreedy(self, d, p):
        # explore (prob = eps)
        if np.random.random()<self._eps(d,p):
            return np.random.choice(self._actions)          
        # exploit (prob = 1 - eps)   
        else:
            i = np.argmax(
                [self._Q[d, p, self._actionMap(a)] for a in self._actions]
                )
            return self._actions[i]
        
    def _actionMap(self, action):
        # map action to numbers: stick=0; hit=1
        assert(action in self._actions), 'Unrecognized Action!'
        if action == 'stick':
            return 0
        else:
            return 1
        
    def _actionMapReverse(self, index):
        assert(index in [0,1]), 'Unrecognized Action Index!'
        if index == 0:
            return 'stick'
        else:
            return 'hit'
        
    def runMCC(self, episodes=1000000, print_freq=100000, report=True):
        
        """
        run MC control
        """       
        
        n_win = 0
        
        for i in range(episodes):           
            episode = [] # list of episode info
            
            # initialize a new game
            game = self._env 
            state = game.initGame()
            
            # sample an episode
            while not game.isTerminated():                
                action = self._epsGreedy(state[0],state[1])
                self._Nsa[state[0], state[1], self._actionMap(action)] += 1
                next_state, reward = game.step(state[1], action)
                episode.append([state, action, reward])
                state = next_state
                       
            # update value function
            for t in range(len(episode)):
                Gt = sum(ep[2] for ep in episode[t:])
                d = episode[t][0][0]
                p = episode[t][0][1]              
                a = self._actionMap(episode[t][1])
                self._Q[d,p,a] += self._alpha(d,p,a) * (Gt-self._Q[d,p,a])
                
            # report performance
            if report:
                n_win += reward == 1 
                win_rate = n_win/(1+i)
                if (i+1)%print_freq==0:
                    p = 'The ' + \
                        '{}'.format(i+1) + \
                        'th episode: Agent won ' + \
                        '{}'.format(n_win) + \
                        ' times | ' + \
                        '{0:.2%}'.format(win_rate) + \
                        ' winning rate'
                    print('\n')
                    print(p)
                
    def getQ(self):
        return self._Q

    def getOptV(self):
        '''
        return optimal value function
        '''
        return self._Q.max(axis=2)
    
    def getOptP(self):
        '''
        return optimal policy
        '''
        Pi = self._Q.argmax(axis=2)
        P = []
        for i in range(Pi.shape[0]):
            x = []
            for j in range(Pi.shape[1]):
                if i==0 or j==0:
                    x.append(None)
                else:
                    x.append(self._actionMapReverse(Pi[i,j]))
            P.append(x)
        P = pd.DataFrame(P)
        return P
     
    def plotV(self):
        Vstar = []
        # reconstruct V* into DataFrame
        v = self.getOptV()
        for d in range(1,11):
            for p in range(1,22):
                Vstar.append([d,p,v[d,p]])
        Vstar = pd.DataFrame(Vstar,columns=['dealer','player','value'])
        
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(Vstar['dealer'],
                        Vstar['player'],
                        Vstar['value'],
                        cmap=plt.cm.viridis, 
                        linewidth=0.2) 
        ax.set_xlabel('Dealer Showing')
        ax.set_ylabel('Player Sum')
        ax.set_zlabel('Optimal Value')
        ax.set_title('Easy21: Monte-Corlo Control')
        plt.show()




#%% run
                
if __name__=='__main__':  
              
    easyMC = MC_Control()                
                                    
    easyMC.runMCC()          
                    
    Q = easyMC.getQ()
    
    V = easyMC.plotV()           
    
    P = easyMC.getOptP()









    