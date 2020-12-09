import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
  def __init__(self, num_agents, message_length, num_tasks, vision_radius, hidden_size=128, msg_vector_size=20):
    super().__init__()
    self.num_agents = num_agents
    self.msg_len = message_length
    self.msg_vector_size = msg_vector_size
    self.num_tasks = num_tasks
    self.hidden_size = hidden_size
    self.vision_radius = vision_radius
    self.vision_array_size = (2*self.vision_radius+1)**2

    self.vector_size = self.num_tasks*3 + (2*self.vision_radius+1)**2 + 6 + self.msg_vector_size #(self.num_agents-1) * self.msg_len

    '''
    ** Input Vector **


    Observations:
    # tasks * 3 array
    row_i = (task_x_i, task_y_i, completed_i)
    flattened ^^

    # 1 * 2 array
    (x, y)

    # (2 * vs+1)^2 array
    A_(centered)ij

    # 1 * 1 array
    map size

    Prev Action:
    # 1 or 0 for move/kill

    Agent ID:
    # 1 * 1 current agent_id

    Agent Type:
    # 1 or 0 for Crewmate or Impostor

    Messages:
    # (n - 1) * k

    '''
    
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()

    # Recurrent
    self.message_network = nn.Linear(self.msg_len * (self.num_agents - 1), self.msg_vector_size)
    self.rnn = nn.GRUCell(input_size=self.vector_size, hidden_size=self.hidden_size)
    self.rnn2 = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
    
    # Q-function + Messages
    # Action space is size 9 (move in one of 8 directions, or kill) by num_agents (who to vote off)
    self.action_space = 9*self.num_agents
    # This is what the DIAL paper used for Q function (feed the outputs of the RNN into this)
    self.outputs = nn.Sequential()
    self.outputs.add_module('linear1', nn.Linear(self.hidden_size, self.hidden_size))
    self.outputs.add_module('relu1', nn.ReLU(inplace=True))
    self.outputs.add_module('linear2', nn.Linear(self.hidden_size, self.action_space))
    

  def format_observations(self, obs, messages):
    tasks, loc, vision, map_size, prev_action, agent_id, agent_type, messages = obs
    
    observation_vector = torch.zeros((self.vector_size,), dtype=torch.float64)

    # embed tasks
    # tasks is num_tasks x 3 array
    observation_vector[0:3*self.num_tasks] = tasks.reshape(-1)
      
    # embed loc
    offset = 3 * len(tasks)
    observation_vector[offset:offset + 2] = loc
    
    # embed vision array
    offset += 2
    observation_vector[offset:offset + self.vision_array_size] = vision.reshape(-1)
    
    # embed map_size
    offset += self.vision_array_size
    observation_vector[offset] = map_size
    
    # embed previous action
    offset += 1
    observation_vector[offset] = prev_action
    
    # embed agent_id
    offset += 1
    observation_vector[offset] = agent_id
    
    # embed agent_type
    offset += 1
    observation_vector[offset] = agent_type
    
    # embed messages
    offset += 1
    observation_vector[offset:] = messages.reshape(-1)
    

  def forward(self, data_batch):
    # data_batch (batch_index, features)
    # Get Message Features
    messages = data_batch[:, :, -(self.num_agents-1) * self.msg_len:]
    messages = torch.Tensor(messages)
   	messages = self.message_network(messages)
    messages = self.sigmoid(messages) # (batch_index, message_features)
    
    # Get input embedding
    x = torch.zeros((len(data_batch), self.vector_size), dtype=torch.float64)
    for j in range(len(data_batch)):
      x[j] = self.format_observations(data_batch[j], messages[j])
    
    
    # Memory Network
    x = self.tanh(self.rnn(x))
    x = self.rnn2(x)
    
    # Q-network (action + new_messages)
    
    
    
    
  # Action sampler: Observation -> Action
  def get_action(self, agent, obs):
        # return [action, direction, vote, *message_bits]
        # action \in {0, 1} => {move, kill}
        # direction \in {0, 1, 3, 4, 5, 6, 7}
        # vote \in [0, len(self.world.agents)]
        # message_bits \in [0, 1]^(n * k - k)

      action_length = 3 + (self.num_agents - 1) * self.k
      action = np.random.random(action_length)

      # TEMP:
      action[0] = np.random.randint(0, 2)  # random action
      action[1] = np.random.randint(0, 8)  # ranodom direction

      ids = [e for e in range(0, self.num_agents) if e != agent.id]
      random.shuffle(ids)
      action[2] = ids[0]  # random id

      return action
