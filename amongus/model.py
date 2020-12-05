import torch
import numpy as np
import random

class Model():
    def __init__(self, num_agents, message_length):
        self.num_agents = num_agents
        self.k = message_length

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
