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
    for agent in agents:
      self.set_agent_location(agent.id, agent.location)

  def set_agent_location(self, id, new_loc):
    r, c = self.agents[id].location
    self.map[r, c] = 0.0
    r, c = new_loc
    self.agent_map[r, c] = World.agent_offset + id
  
  def check_task_accomplished(self, location):
    pass
  def do_task(self, location):
    pass
  
  def __repr__(self) -> str:
    return np.array_str(np.maximum(self.map, self.agent_map))
