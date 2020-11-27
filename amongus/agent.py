from amongus.model import Model

class Agent:
  model = Model()
  
  def __init__(self, id, world, vision_radius=3):
    self.id = id
    self.world = world
    self.alive = True
    self.location = (0,0) #change later
    self.vision_radius = vision_radius

  def get_action():
    pass

  def update_location(move_action):
    # 0: up
    # 1: up + right
    # 2: right
    # 3: down + right
    # 4: down
    # 5: down + left
    # 6: left
    # 7: left + up
    # Move, but stay in bounds
    if move_action == 0 or move_action == 1 or move_action == 7:
        self.location[1] = max(0, self.location[1]-1) 
    if move_action >= 1 and move_action <= 3:
        self.location[0] = min(self.world.n-1, self.location[0]+1)
    if move_action >= 3 and move_action <= 5:
        self.location[1] = min(self.world.n-1, self.location[1]+1)
    if move_action >= 5 and move_action <= 7:
        self.location[0] = max(0, self.location[0]-1)
    return self.location
    

class Crewmate(Agent):
  def __init__(self, id, world, vision_radius=3):
    super().__init__(id, world, vision_radius)

class Impostor(Agent):
  def __init__(self, id, world, vision_radius=3, kill_distance=1):
    super().__init__(id, world, vision_radius)
    self.kill_distance = kill_distance