from amongus.agent import Crewmate, Impostor
from amongus.world import World
import numpy as np
import random

class Game:

  # Default Game Parameters
  params = {
    'kill_distance': 1,
    'agent_vision_radius': 3,
    'task_reward': 3,
    'win_reward': 100,
    'kill_reward': 3
  }

  def __init__(self, map_size, num_tasks, num_agents, max_steps, vote_period, params=params):
    self.n = map_size
    self.t = num_tasks
    self.num_agents = num_agents
    self.max_steps = max_steps
    self.vote_period = vote_period
    
    # Game Parameters
    self.kill_distance = params.get('kill_distance', Game.params['kill_distance'])
    self.agent_vision_radius = params.get('agent_vision_radius', Game.params['agent_vision_radius'])
    self.task_reward = params.get('task_reward', Game.params['task_reward'])
    self.win_reward = params.get('win_reward', Game.params['win_reward'])
    self.kill_reward = params.get('win_reward', Game.params['kill_reward'])

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

            if self.agents[to_kill].alive and agent.distance_to(self.agents[to_kill].location) <= self.kill_distance:
                to_kill = action[2] # We migh want to just kill whoever is within distance for the impostor ?

        elif action[0] == 1:
            # We move, we move
            task_done = self.world.move_agent(agent.id, action[1])
            if task_done:
                rewards[agent.id] = self.task_reward
    
    # Check task win condition
    if self.world.tasks_left <= 0:
        add_rewards = self.win_reward * np.ones(self.num_agents)
        add_rewards[self.impostor_index] = -self.win_reward # Or 0?
        rewards = rewards + add_rewards # I think this does what I want it to...
        self.ongoing = False
        # Return something
    
    # Kill someone
    if to_kill is not None:
        self.world.kill_agent(to_kill)
        dead_agent = self.agent[to_kill]
        dead_agent.alive = False
        r,c = dead_agent.location
        self.world.agent_map[r,c] = 0.0
        rewards[self.impostor_index] = self.kill_reward
    
    # Check kill win condition
    if self.world.num_agents <= 2:
        add_rewards = -self.win_reward * np.ones(self.num_agents) # Or 0?
        add_rewards[self.impostor_index] = self.win_reward
        rewards = rewards + add_rewards # I think this does what I want it to...
        self.ongoing = False
        # Return something
    
    # TODO: voting someone out and winning based off that
    if (self.num_steps + 1) % self.vote_period == 0:
        # Perform voting
    
    self.num_steps += 1
    if self.num_steps >= self.max_steps:
        self.ongoing = False

  def __repr__(self) -> str:
    world_arr = np.array(np.maximum(self.world.map, self.world.agent_map), dtype=object)
    
    def make_blank(x):
      if x == 0.0:
        return ''
      elif x == 1.0:
        return 'G' # Goal Location
      else:
        return 'A{:^d}'.format(int(x - World.agent_offset))

    make_blank = np.vectorize(make_blank)
    world_arr = make_blank(world_arr)

    out = '-' * (6 * len(world_arr[0]) + 2) + '\n'
    for row in world_arr:
      fstring = '{:^5s}|' * len(row)
      out += '|' + fstring[:-1].format(*row) + '|' + '\n'
      out += '-' * (6 * len(row) + 2) + '\n'
    return out