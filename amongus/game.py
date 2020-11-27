from amongus.agent import Crewmate, Impostor
from amongus.world import World
import numpy as np
import random

class Game:

  # Default Game Parameters
  params = {
    'kill_distance': 1,
    'agent_vision_radius': 3,
    'task_reward': 3
  }

  def __init__(self, map_size, num_tasks, num_agents, max_steps, params=params):
    self.n = map_size
    self.t = num_tasks
    self.num_agents = num_agents
    self.max_steps = max_steps
    
    # Game Parameters
    self.kill_distance = params.get('kill_distance', Game.params['kill_distance'])
    self.agent_vision_radius = params.get('agent_vision_radius', Game.params['agent_vision_radius'])
    self.task_reward = params.get('task_reward', Game.params['task_reward'])

    self.reset()

  def reset(self):
    # Generate a new world
    self.world = World(self.n, self.t)

    # Pick num_agents different map locations
    rng = np.random.default_rng()
    locs = rng.choice(self.n * self.n, size=self.num_agents, replace=False)
    coord_map = np.vectorize(lambda x: (x//self.n, x%self.n))
    locs = np.vstack(coord_map(locs)).T

    # Instatntiate num_agents - 1 Crewmates and 1 Impostor agents
    self.impostor_index = random.randint(0, self.num_agents-1)
    self.agents = []
    for i in range(self.num_agents):
        if i == self.impostor_index:
            self.agents.append(Impostor(i, self.world, locs[i], vision_radius=self.agent_vision_radius, kill_distance=self.kill_distance))
        else:
            self.agents.append(Crewmate(i, self.world, locs[i], vision_radius=self.agent_vision_radius))
    self.world.set_agent_list(self.agents)

    # Whether the game is ongoing
    self.ongoing = True
    self.num_steps = 0
  
  def step(self):
    # This function returns the new observation for each agent,
    # The reward they got,
    # Whether the game is done 
    to_kill = None
    rewards = np.zeroes(self.num_agents)
    
    for agent in self.agents:
        # Dead agents cannot do anything
        if not agent.alive:
            continue 

        action = agent.get_action()
        # Assume action to be a row vector 
        # Index 0 represents kill(0) or move(1)
        # Index 1 represents which direction to move
        # Index 2 represents who to kill
        # Remainder of the k*(n-1) indices represent messages to each other agent

        if action[0] == 0:
            # Only the impostor can kill someone else 
            if agent.id != self.impostor_index:
                continue

            # We are killing this person, they die in the next round
            to_kill = action[2] # We migh want to just kill whoever is within distance for the impostor ?
            can_kill = agent.distance_to(self.agents[to_kill]) <= self.kill_distance

            if can_kill:
                pass
                # rewards[agent]

            else:
                to_kill = None

        elif action[0] == 1:
            # We are moving
            new_location = agent.update_location(action[1])
            task_done = self.world.check_task_accomplished(new_location)
            if task_done:
                rewards[agent.id] = self.task_reward    
    
    self.num_steps += 1
    if self.num_steps >= self.max_steps:
        self.ongoing = False
        

  def get_observation(self, agent):
    # Get a 2*vision_radius+1 size square representing what the agent can see
    pass

  def generate_agent_rewards(self, agent):
    # If the agent is an crewmate in the square and the task was just done,
    # award the agent for doing the task
    # If the agent is an impostor and they just killed someone, award the agent 
    # for killing them
    
    pass
  
  def generate_rewards(self):
    rewards = []
    for agent in self.agents:
        rewards.append(generate_agent_rewards(agent))
    return np.array(rewards)

  def transition(self, actions):
    # Transition the world state, where actions is 
    # an array of the actions of each agent 
    # Actions are move or kill 
    
    pass

  def __repr__(self) -> str:
    world_arr = np.array(np.maximum(self.world.map, self.world.agent_map), dtype=object)
    
    def make_blank(x):
      if x == 0.0:
        return ''
      elif x == 1.0:
        return 'G'
      else:
        return 'A{:^1d}'.format(int(x - World.agent_offset))

    make_blank = np.vectorize(make_blank)
    world_arr = make_blank(world_arr)

    out = '-' * (6 * len(world_arr[0]) + 2) + '\n'
    for row in world_arr:
      fstring = '{:^5s}|' * len(row)
      out += '|' + fstring[:-1].format(*row) + '|' + '\n'
      out += '-' * (6 * len(row) + 2) + '\n'
    return out