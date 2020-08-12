#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:36:18 2019

@author: victor
"""

#%% import libs

import numpy as np
import pandas as pd
 
from Environment import Easy21
from Monte_Carlo_Control import MC_Control

import matplotlib.pyplot as plt


#%% TD Learning

class TD_Control(object):
    """
    
    Implement Sarsa(λ) in 21s. Initialise the value function to zero. Use the 
    same step-size and exploration schedules as in the MC section. Run 
    the algorithm with parameter values λ ∈ {0, 0.1, 0.2, ..., 1}. Stop each 
    run after 1000 episodes and report the mean-squared error 
    􏰀sum(Q(s, a) − Q∗(s, a))**2 over all states s and actions a, comparing the 
    true values Q∗(s,a) computed in the previous section with the estimated 
    values Q(s, a) computed by Sarsa. Plot the mean-squared error against λ. 
    For λ = 0 and λ = 1 only, plot the learning curve of mean-squared error
    against episode number.
    
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
        
    def Sarsa(self, 
              lambda_=0, 
              episodes=1000, 
              learningRate = False,
              Qstar = None):
        """
        Sarsa(λ), backward view
        """
        
        # initialize learning rates
        if learningRate and Qstar is not None:
            lr = []
        
        for i in range(episodes):
            
            # list of episode info
            episode = [] 
            
            # initialize eligibility trace
            E = np.zeros((11,22,2))
            
            # initialize a game & action
            game = self._env
            state = game.initGame()
            action = self._epsGreedy(state[0], state[1])
            
            # sample trajectory
            while not game.isTerminated():
                
                a = self._actionMap(action)
                episode.append([state[0],state[1],a])
                self._Nsa[state[0], state[1], a] += 1
                
                # take action, observe next state and reward
                next_state, reward = game.step(state[1], action)
                
                if not game.isTerminated():
                    
                    # sample next action
                    next_action = self._epsGreedy(next_state[0], next_state[1]) 
                    next_a = self._actionMap(next_action)
                    
                    # update TD error
                    delta = reward + \
                            self._Q[next_state[0], next_state[1], next_a] - \
                            self._Q[state[0], state[1], a]
                            
                # if terminated, no next action, Q[ternimate] = 0           
                else:
                    delta = reward - self._Q[state[0], state[1], a]
                        
                # update eligibility trace for current S and A       
                E[state[0], state[1], a] += 1
                
                # update Q and decay eligibility trace for all experience 
                # of current episode
                for d,p,a in episode:
                    self._Q[d,p,a] += self._alpha(d,p,a) * delta * E[d,p,a]
                    E[d,p,a] *= lambda_
    
                # update state and action
                if not game.isTerminated():
                    state = next_state
                    action = next_action
                    
            # append learning rates
            if learningRate and Qstar is not None:
                error = Qstar - self._Q    
                mse = (error**2).sum()/(21*10*2)
                lr.append(mse)
        
        if learningRate and Qstar is not None:
            return lr
        
        
        
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
        ax.set_title('Easy21: Temporal-Difference Control')
        plt.show()
        
        
        
        
        
#%% run

if __name__=='__main__': 
    
    easyMC = MC_Control()
    easyMC.runMCC()
    Q_MC = easyMC.getQ()
    
    fig = plt.figure()
    
    # plot mean square error
    ax1 = fig.add_subplot(2,1,1) 
    
    mse_list = []
    for l in np.arange(0, 1.1, 0.1):               
        easyTD = TD_Control()            
        easyTD.Sarsa(lambda_=l, episodes=1000)     
        Q_TD = easyTD.getQ()
        error = Q_MC - Q_TD    
        mse = (error**2).sum()/(21*10*2)
        mse_list.append(mse)
              
    plt.plot(np.arange(0, 1.1, 0.1), mse_list)        
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.title('Sarsa: MSE for each lambda')
    
    # plot learning rate
    ax2 = fig.add_subplot(2,1,2)
    for l in np.arange(0, 1.1, 0.1):               
        easyTD = TD_Control()            
        LR_TD = easyTD.Sarsa(lambda_=l,
                             episodes=1000,
                             learningRate=True,
                             Qstar=Q_MC)     
        plt.plot(LR_TD, linewidth=1)  
    plt.xlabel('Episodes')
    plt.ylabel('MSE')
    plt.title('Sarsa: MSE for each episode')        
    plt.legend(np.arange(0, 11)/10, fontsize=5)
        
    plt.tight_layout()
        