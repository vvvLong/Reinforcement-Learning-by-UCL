#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:42:38 2019

@author: victor
"""

#%% import libs

import numpy as np


#%% Environment

class Easy21(object):
    """
    
    Rules:
        
    • The game is played with an infinite deck of cards 
      (i.e. cards are sampled with replacement)
      
    • Each draw from the deck results in a value between 1 and 10 
      (uniformly distributed) with a colour of red (probability 1/3) or black 
      (probability 2/3)
      
    • There are no aces or picture (face) cards in this game
    
    • At the start of the game both the player and the dealer draw one black
      card (fully observed)
      
    • Each turn the player may either stick or hit
    
    • If the player hits then she draws another card from the deck
    
    • If the player sticks she receives no further cards
    
    • The values of the player’s cards are added (black cards) or 
      subtracted (red cards)
      
    • If the player’s sum exceeds 21, or becomes less than 1, then she 
      “goes bust” and loses the game (reward -1)
      
    • If the player sticks then the dealer starts taking turns. The dealer 
      always sticks on any sum of 17 or greater, and hits otherwise. If the 
      dealer goes bust, then the player wins; otherwise, the outcome – win 
      (reward +1), lose (reward -1), or draw (reward 0) – is the player with 
      the largest sum.
      
    ---------------------------------------------------------------------------  
      
    Implementation:
        
    Specifically, write a function, named step, which takes as input a state s 
    (dealer’s first card 1–10 and the player’s sum 1–21), and an action a (hit 
    or stick), and returns a sample of the next state s′ (which may be terminal 
    if the game is finished) and reward r. We will be using this environment 
    for model-free reinforcement learning, and you should not explicitly 
    represent the transition matrix for the MDP. There is no discounting 
    (γ = 1). You should treat the dealer’s moves as part of the environment, 
    i.e. calling step with a stick action will play out the dealer’s cards and 
    return the final reward and terminal state.
      
    """
    
    def __init__(self):
        self._card_min = 1
        self._card_max = 10
        self._black_ratio = 2/3
        
        self._game_min = 1
        self._game_max = 21
        
        self._dealer_threshold = 17
        self._dealer_first = None # dealer's first card
        self._dealer_val = None # dealer's overall value
        
        self._action_space = ['stick', 'hit']
        
        self._terminated = True # check if a game is terminated/not started 
        
    def initGame(self):
        assert(self._terminated), 'The game has been initialized already!'
            
        self._terminated = False
        
        #print('\n')
        #print('A new game has started!')
        
        # card for dealer
        card = np.random.randint(self._card_min, self._card_max+1)
        self._dealer_first = card
        self._dealer_val = card
        
        # card for agent
        agent = np.random.randint(self._card_min, self._card_max+1)
        return (self._dealer_first, agent)

    def _draw(self):
        """
        red = -1
        black = 1
        """
        card = np.random.randint(self._card_min, self._card_max+1)        
        color = 2 * ( (np.random.random()<self._black_ratio) - 0.5 )
        return card*color
    
    def _dealerLoop(self):
        while self._dealer_val >= self._game_min and \
              self._dealer_val < self._dealer_threshold:
            self._dealer_val += self._draw()
            
    def _isBusted(self, value):
        return value>self._game_max or value<self._game_min
    
    def step(self, agent_val, action):
        
        assert(action in self._action_space), 'Unrecognized Action!'
        
        assert(not self._terminated), "New game hasn't been initialized yet!" 
        
        # agent to stick
        if action == 'stick':
            agent = agent_val
            self._dealerLoop()
            
            # when dealer goes bust or agent has larger value
            if self._isBusted(self._dealer_val) or agent > self._dealer_val:
                reward = 1
                self._terminated = True # game is terminated
                
            elif agent == self._dealer_val:
                reward = 0
                self._terminated = True
            
            else:
                reward = -1
                self._terminated = True
            
        # agent to hit    
        else:
            agent = agent_val + self._draw()
            
            # agent goes bust
            if self._isBusted(agent):
                reward = -1
                self._terminated = True
            
            # game to continue
            else:
                reward = 0

        # output state
        next_state = (self._dealer_first, int(agent))
        
        # clear memory and print if ternimated
        if self._terminated:
            self._dealer_first = None 
            self._dealer_val = None 
            #print('\n')
            #print('The game is terminated!')
        
        return next_state, reward
    
    def actionSpace(self):
        return self._action_space
    
    def isTerminated(self):
        return self._terminated
        

            
    
#%% run 

if __name__== "__main__":
    
    game = Easy21()        
        
    agent = game.initGame()  
        
    state, reward = game.step(agent,'hit')     
    agent = state[1]
    game._dealer_val    
        
    state, reward = game.step(agent,'stick')     
    

    
    
    
    
       