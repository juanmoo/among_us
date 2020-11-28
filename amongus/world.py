import numpy as np

class World:

  '''
  Map Key

  0 -> Empty Spot
  1 -> Task Location

  n >= 100:
  n -> Agent - 100 currently occupies that square
  '''
  agent_offset = 100

  def __init__(self, n, num_tasks):
    # Initialize nxn map 
    self.n = n
    self.num_tasks = num_tasks
    self.tasks_left = num_tasks
    
    # Generate blank map
    world_map = np.zeros(n*n)
    self.agent_map = np.zeros((n, n))

    # Add tasks at front
    world_map[:num_tasks] = 1
    # Randomly shuffle the tasks in 
    np.random.shuffle(world_map)
    # Make map the right dimensions
    world_map = world_map.reshape((n, n))
    self.map = world_map

  def set_agent_list(self, agents):
    self.agents = agents
    self.num_agents = len(agents)
    for agent in agents:
      r, c = agent.location
      self.agent_map[r, c] = World.agent_offset + agent.id

  def move_agent(self, id, move_action):
    r, c  = self.agents[id].location
    self.agent_map[r, c] = 0.0
    r, c = self.agents[id].update_location(move_action)
    self.agent_map[r, c] = World.agent_offset + id
    # Check if agent is a crewmate and that there is a task at the new location
    if self.map[r, c] == 1.0 and self.agents[id].agent_type == 'Crewmate':
      self.tasks_left -= 1
      self.map[r, c] == 0.0
      return True
    return False

  def kill_agent(self, id):
    dead_agent = self.agents[to_kill]
    dead_agent.alive = False
    r,c = dead_agent.location
    self.agent_map[r,c] = 0.0
    self.agents[id].alive = False
    self.num_agents -= 1
  
  def __repr__(self) -> str:
    return np.array_str(np.maximum(self.map, self.agent_map))
