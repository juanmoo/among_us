class Game:
  def __init__(self, map_size, num_tasks, num_agents, max_steps):
    self.n = map_size
    self.t = num_tasks
    self.num_agents = num_agents
    self.max_steps = max_steps
    #Arbitrary numbers
    self.kill_distance = 1
    self.agent_vision_radius = 3
    self.task_reward = 3
    self.reset()

  def reset(self):
    # Generate a new world
    self.world = World(self.n, self.t)
    # Make all but one crewmates
    self.impostor_index = random.randint(0, self.num_agents-1)
    self.agents = []
    for i in range(self.num_agents):
        if i == self.impostor_index:
            self.agents.append(Impostor(i, self.world, self.agent_vision_radius, self.kill_distance))
        else:
            self.agents.append(Crewmate(i, self.world, self.agent_vision_radius))
    # Whether the game is ongoing
    self.ongoing = True
    self.num_steps = 0
  
  def step(self):
    # This function returns the new observation for each agent,
    # The reward they got,
    # Whether the game is done 
    to_kill = None
    rewards = np.zeroes(num_agents)
    
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
            to_kill = action[2]
            can_kill = check_proximity(to_kill) #implement checking that the person is close enough to kill
            if can_kill:
                pass
                # rewards[agent]
            else:
                to_kill = None
        if action[0] == 1:
            # We are moving
            new_location = agent.update_location(action[1])
            task_done = self.world.check_task_accomplished(new_location)
            if task_done:
                rewards[agent.id] = self.task_reward    
    
    self.num_steps += 1
    if self.num_steps >= self.max_steps:
        self.ongoing = False
        
    pass

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