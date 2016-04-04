
# Train a Smartcab How to Drive

Reinforcement Learning (Q learning) Project

##Overview
The following description/overview was provided by Udacity for this project
>###Train a Smartcab to Drive

>A smartcab is a self-driving car from the not-so-distant future that ferries people from one arbitrary location to another. In this project, you will use reinforcement learning to train a smartcab how to drive.

>###Environment

>Your smartcab operates in an idealized grid-like city, with roads going North-South and East-West. Other vehicles may be present on the roads, but no pedestrians. There is a traffic light at each intersection that can be in one of two states: North-South open or East-West open.

>US right-of-way rules apply: On a green light, you can turn left only if there is no oncoming traffic at the intersection coming straight. On a red light, you can turn right if there is no oncoming traffic turning left or traffic from the left going straight.

>To understand how to correctly yield to oncoming traffic when turning left, you may refer to this [official drivers’ education video](https://www.youtube.com/watch?v=TW0Eq2Q-9Ac), or this [passionate exposition](https://www.youtube.com/watch?v=0EdkxI6NeuA).

>###Inputs

>Assume that a higher-level planner assigns a route to the smartcab, splitting it into waypoints at each intersection. And time in this world is quantized. At any instant, the smartcab is at some intersection. Therefore, the next waypoint is always either one block straight ahead, one block left, one block right, one block back or exactly there (reached the destination).

>The smartcab only has an egocentric view of the intersection it is currently at (sorry, no accurate GPS, no global location). It is able to sense whether the traffic light is green for its direction of movement (heading), and whether there is a car at the intersection on each of the incoming roadways (and which direction they are trying to go).

>In addition to this, each trip has an associated timer that counts down every time step. If the timer is at 0 and the destination has not been reached, the trip is over, and a new one may start.## Install

>This project requires Python 2.7 with the pygame library installed:

>[https://www.pygame.org/wiki/GettingStarted](https://www.pygame.org/wiki/GettingStarted)

>###Outputs

>At any instant, the smartcab can either stay put at the current intersection, move one block forward, one block left, or one block right (no backward movement).

>###Rewards

>The smartcab gets a reward for each successfully completed trip. A trip is considered “successfully completed” if the passenger is dropped off at the desired destination (some intersection) within a pre-specified time bound (computed with a route plan).

>It also gets a smaller reward for each correct move executed at an intersection. It gets a small penalty for an incorrect move, and a larger penalty for violating traffic rules and/or causing an accident.

>###Goal

>Design the AI driving agent for the smartcab. It should receive the above-mentioned inputs at each time step t, and generate an output move. Based on the rewards and penalties it gets, the agent should learn an optimal policy for driving on city roads, obeying traffic rules correctly, and trying to reach the destination within a goal time.

## Setup and Packages

>###Setup
>You need Python 2.7 and pygame for this project: https://www.pygame.org/wiki/GettingStarted
For help with installation, it is best to reach out to the pygame community [help page, Google group, reddit].

###Programs & Packages Required
- Pygame
- Python 2.7
- Numpy (python package)
- Pandas (python package)
- Itertools (python package)
- copy (python package)
- time (python package)
- random (python package)
- os (python package)
- pygame (python package)
- iPython Notebook (with iPython 4.0)
 
###File Descriptions
- environment.py: This is a class written by Udacity that creates the city grid and traffic rewards/penalties
- planner.py: This class was written by Udacity and it provides a default next_waypoint for an agent if called
- simulation.py: This class was written by Udacity and allows the user to visually see the agent(s) moving in the city grid
- agent.py: The shell of this file was provided by Udacity, but the guts of the learning agent logic were created by me.  See commit history to tell the difference between the Udacity provided file and the finished product.
 
## Run

Fork/clone this repository to your local computer.  Make sure you are in the top-level project directory `smartcab/` (that contains this README). Then run:

```python smartcab/agent.py```

OR:

```python -m smartcab.agent```

##MIT License

Copyright (c) 2016 Jed Isom

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
