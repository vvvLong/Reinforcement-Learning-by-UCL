#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 20:49:44 2019

@author: victor
"""

#%% import libs

import numpy as np
import pandas as pd
 
from Environment import Easy21
from Monte_Carlo_Control import MC_Control

import matplotlib.pyplot as plt


#%% Linear Function Approximation with TD

class LFA_Control(object):
    """
    
    Use a binary feature vector φ(s, a) with 3 ∗ 6 ∗ 2 = 36 features. Each 
    binary feature has a value of 1 iff (s, a) lies within the cuboid of 
    state-space corresponding to that feature, and the action corresponding 
    to that feature. The cuboids have the following overlapping intervals:
        
        dealer(s) = {[1, 4], [4, 7], [7, 10]}
        player(s) = {[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]}
        a = {hit, stick}
        
        where
            • dealer(s) is the value of the dealer’s first card (1–10) 
            • sum(s) is the sum of the player’s cards (1–21)
            
    Repeat the Sarsa(λ) experiment from the previous section, but using linear 
    value function approximation Q(s, a) = φ(s, a)⊤θ. Use a constant 
    exploration of ε = 0.05 and a constant step-size of 0.01. Plot the 
    mean-squared error against λ. For λ = 0 and λ = 1 only, plot the learning 
    curve of mean-squared error against episode number.
    
    """
    
    def __init__(self, N0=100):
        self._env = Easy21()
        self._actions = self._env.actionSpace()
        
        # linear approx. function and parameters
        self._w = np.zeros([36,1]) 
        self._Q = lambda d,p,a: np.dot(self._feature(d,p,a).T, self._w) 
         
        self._eps = 0.05
        self._alpha = 0.01
        
    def _epsGreedy(self, d, p):
        # explore (prob = eps)
        if np.random.random()<self._eps:
            return np.random.choice(self._actions)          
        # exploit (prob = 1 - eps)   
        else:
            i = np.argmax([self._Q(d, p, a)] for a in self._actions)
            return self._actions[i]
         
    def _feature(self, d, p, action):
        feature = np.zeros([3,6,2])
        a = self._actionMap(action)
        
        # insert feature 
        for i,(lowerd,upperd) in enumerate(zip(range(1,8,3),range(4,11,3))):
            for j,(lowerp,upperp) in enumerate(zip(range(1,17,3),range(6,22,3))):
                feature[i,j,:] = lowerd <= d <= upperd and lowerp <= p <= upperp     
        if a==0:
            feature[:,:,1] = 0
        else:
            feature[:,:,0] = 0       
        return feature.reshape(-1,1)
         
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
              learningRate=False,
              Qstar=None):
        """
        Sarsa(λ), backward view
        """
        
        # initialize learning rates
        if learningRate and Qstar is not None:
            lr = []
        
        for i in range(episodes):
            
            # initialize eligibility trace
            # be very careful here, not corresponding to s,a anymore 
            E = np.zeros([36,1])
            
            # initialize a game & action
            game = self._env
            state = game.initGame()
            action = self._epsGreedy(state[0], state[1])
            
            # sample trajectory
            while not game.isTerminated():
                
                # take action, observe next state and reward
                next_state, reward = game.step(state[1], action)
                
                if not game.isTerminated():
                    
                    # sample next action
                    next_action = self._epsGreedy(next_state[0], next_state[1]) 
                    
                    # update TD error
                    delta = reward + \
                            self._Q(next_state[0],next_state[1],next_action) - \
                            self._Q(state[0], state[1], action)
                           
                # if terminated, no next action, Q[ternimate] = 0           
                else:
                    delta = reward - self._Q(state[0], state[1], action)
                        
                # update eligibility trace       
                E = lambda_ * E + self._feature(state[0],state[1],action)
                
                # update w
                self._w += self._alpha * delta * E
    
                # update state and action
                if not game.isTerminated():
                    state = next_state
                    action = next_action
                    
            # append learning rates
            if learningRate and Qstar is not None:
                error = Qstar - self.getQVal()    
                mse = (error**2).sum()/(21*10*2)
                lr.append(mse)
        
        if learningRate and Qstar is not None:
            return lr
        
        
        
    def getQFun(self):
        return self._Q
    
    def getQVal(self):
        Q = np.zeros([11,22,2])
        for d in range(11):
            for p in range(22):
                for a in range(2):
                    action = self._actionMapReverse(a)
                    Q[d,p,a] = self._Q(d,p,action)
        return Q

    def getOptV(self):
        '''
        return optimal value function
        '''
        return self.getQVal().max(axis=2)
    
    def getOptP(self):
        '''
        return optimal policy
        '''
        Pi = self.getQVal().argmax(axis=2)
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
        easyLFA = LFA_Control()            
        easyLFA.Sarsa(lambda_=l, episodes=1000)     
        Q_LFA = easyLFA.getQVal()
        error = Q_MC - Q_LFA    
        mse = (error**2).sum()/(21*10*2)
        mse_list.append(mse)
              
    plt.plot(np.arange(0, 1.1, 0.1), mse_list)        
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.title('MSE: LFA for each lambda')
    
    # plot learning rate
    ax2 = fig.add_subplot(2,1,2)
    for l in np.arange(0, 1.1, 0.1):               
        easyLFA = LFA_Control()            
        LR_LFA = easyLFA.Sarsa(lambda_=l,
                             episodes=10000,
                             learningRate=True,
                             Qstar=Q_MC)     
        plt.plot(LR_LFA, linewidth=1)  
    plt.xlabel('Episodes')
    plt.ylabel('MSE')
    plt.title('MSE: LFA for each episode')        
    plt.legend(np.arange(0, 11)/10, fontsize=5)
        
    plt.tight_layout()