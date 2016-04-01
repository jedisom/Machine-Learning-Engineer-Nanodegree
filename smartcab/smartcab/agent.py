import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        deadline = self.env.get_deadline(self) #Current dealine remaining (in time steps)        
        inputs = self.env.sense(self)  #Need to determine traffic light status, presence of cars
        state = self.env.agent_states[self]
        Xdist = self.planner.destination[0] - state['location'][0]  #Get delta X distance
        Ydist = self.planner.destination[1] - state['location'][1]  #Get delta Y distance        
        heading = state['heading']

        # TODO: Update state
        
        
        # TODO: Select action according to your policy
        # Part I of assignment tells me to randomly select an action        
        action = random.choice(self.env.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

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
