import random
import numpy as np
import pandas as pd
import itertools
from environment import Agent, Environment, TrafficLight
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.gamma = 0.5 #discount rate
        self.alpha = 0.5 #learning rate
    
        #CREATE STARTING Q LEARNING MATRIX
        #Calculate variables determining the matrix' size
        actions = self.env.valid_actions
        lights = TrafficLight().valid_states        
        Xlist = np.arange(self.env.grid_size[0])
        Ylist = np.arange(self.env.grid_size[1])                
        headings = self.env.valid_headings
        #self.T = 0   
        
        #create Q matrix
        # rows based on this coding system (next_waypoint, light_status, oncoming, left, right, Xdist,
        #                                   Ydist, heading, #steps 'til deadline, ) 
        #only do this once and write it to csv file to be read later   
        if (os.path.isfile("row_names.csv") == False):
            c = csv.writer(open("row_names.csv", "wb"))            
            row_names = np.empty(0)       
            for combination in itertools.product(actions, lights, actions, actions, actions, Xlist, Ylist, headings): #T, 
                #initialize as zeros            
                row_name = ', '.join(map(str, list(combination)))          
                c.writerow(row_name)                
                row_names = np.append(row_names,row_name)
            
            
        self.Q = pd.DataFrame(index=row_names, columns=actions)
        self.Q = self.Q.fillna(0)
        self.Qprime = self.Q
        print "DONE; start learning..."



    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        deadline = self.env.get_deadline(self) #Current dealine remaining (in time steps)        
        inputs = self.env.sense(self)  #Need to determine traffic light status, presence of cars
        state = self.env.agent_states[self]
        
        #May want to rework these distance measures because the map wraps around
        Xdist = self.planner.destination[0] - state['location'][0]  #Get delta X distance
        Ydist = self.planner.destination[1] - state['location'][1]  #Get delta Y distance        
        heading = state['heading']

        # TODO: Update state
        # state rows based on this coding system (next_waypoint, light_status, oncoming, left, right, Xdist,
        #                                         Ydist, heading, #steps 'til deadline, )         
        current_state = ', '.join(map(str, list(self.next_waypoint, inputs, Xdist, Ydist, heading))) #time left to reach desination
        
        # TODO: Select action according to your policy
        # Part I of assignment tells me to randomly select an action        
        action = random.choice(self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        #find maximum Qprime over all actions
        max_Qprime = ???
        utility = reward + self.gamma * max_Qprime
        self.Q = (1-self.alpha)*self.Q + self.alpha(utility)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, Xdist = {}, Ydist = {}, heading = {}, reward = {}".format(deadline, inputs, action, Xdist, Ydist, heading, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.5)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
