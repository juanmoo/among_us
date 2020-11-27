import numpy as np

class World:
  def __init__(self, n, num_tasks):
    # Initialize nxn map 
    self.n = n
    self.num_tasks = num_tasks
    
    # Generate blank map
    world_map = np.zeros(n*n)
    # Add tasks at front
    world_map[:num_tasks] = 1
    # Randomly shuffle the tasks in 
    np.random.shuffle(world_map)
    # Make map the right dimensions
    world_map = world_map.reshape((n, n))
    
    self.map = world_map

  def __repr__(self) -> str:
    return np.array_str(self.map)

  def check_task_accomplished(self, location):
    pass
  def do_task(self, location):
    pass
