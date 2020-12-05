from amongus.model import Model
import numpy as np
import random


class Agent:

    def __init__(self, id, world, start_location, vision_radius=3, message_length=1):
        self.id = id
        self.world = world
        self.alive = True
        self.location = start_location
        self.vision_radius = vision_radius
        self.message_length = message_length
        self.agent_type = 'Undefined'

    def distance_to(self, loc):
        return np.linalg.norm(self.location - loc)

    def sense_agents(self):
        agents_seen = []  # (Agent_id, loc)
        cr, cc = self.location
        vr = int(self.vision_radius)
        for i in range(-1 * vr, vr + 1):
            for j in range(-1 * vr, vr + 1):
                in_map_r = (0 <= cr + i < self.world.n)
                in_map_c = (0 <= cc + j < self.world.n)
                in_map = in_map_r and in_map_c
                if in_map and i**2 + j**2 <= self.vision_radius**2:
                    reading = self.world.agent_map[cr + i, cc + j]
                    if reading >= 100:
                        agents_seen.append(
                            self.world.agents[int(reading - 100)])

        # Sort by distance
        agents_seen.sort(key=lambda a: self.distance_to(a.location))
        return agents_seen[1:]

    def get_observation(self):
        base_world = np.copy(self.world.map)
        mask = np.zeros((self.world.n, self.world.n))
        vr = int(self.vision_radius)
        cr, cc = self.location
        for i in range(-1 * vr, vr + 1):
            for j in range(-1 * vr, vr + 1):
                in_map_r = (0 <= cr + i < self.world.n)
                in_map_c = (0 <= cc + j < self.world.n)
                in_map = in_map_r and in_map_c
                if in_map and i**2 + j**2 <= self.vision_radius**2:
                    mask[cr + i, cc + j] = 1.0

        obs = (base_world, self.world.agent_map * mask)

        return obs

    # TODO: need to remove decision making from agent
    def get_action(self):
        # return [action, direction, vote, *message_bits]
        # action \in {0, 1} => {move, kill}
        # direction \in {0, 1, 3, 4, 5, 6, 7}
        # vote \in [0, len(self.world.agents)]
        # message_bits \in [0, 1]^(n * k - k)

        action_length = 3 + (len(self.world.agents) - 1) * self.message_length
        # action = np.array([-1] * action_length, dtype=np.float64)
        action = np.random.random(action_length)

        # TEMP:
        action[0] = np.random.randint(0, 2)  # random action
        action[1] = np.random.randint(0, 8)  # ranodom direction

        ids = [e for e in range(0, len(self.world.agents)) if e != self.id]
        random.shuffle(ids)
        action[2] = ids[0]  # random id

        return action

    def update_location(self, move_action):
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

    def __repr__(self) -> str:
        return 'ID: {}, Role: {}, Location: ({}, {})'.format(self.id, self.agent_type, *self.location)


class Crewmate(Agent):
    def __init__(self, id, world, start_location, message_length=1, vision_radius=3):
        super().__init__(id, world, start_location, message_length, vision_radius)
        self.agent_type = 'Crewmate'


class Impostor(Agent):
    def __init__(self, id, world, start_location, vision_radius=3, message_length=1, kill_distance=1):
        super().__init__(id, world, start_location, message_length, vision_radius)
        self.agent_type = 'Impostor'
        self.kill_distance = kill_distance
