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
        'kill_reward': 3,
        'message_length': 1,
        'impostor_frac': 0.1,
        'vote_period': 10
    }

    def __init__(self, map_size, num_tasks, num_agents, max_steps, params=params):
        self.n = map_size
        self.t = num_tasks
        self.num_agents = num_agents
        self.max_steps = max_steps

        # Game Parameters
        self.kill_distance = params.get(
            'kill_distance', Game.params['kill_distance'])
        self.agent_vision_radius = params.get(
            'agent_vision_radius', Game.params['agent_vision_radius'])
        self.task_reward = params.get(
            'task_reward', Game.params['task_reward'])
        self.win_reward = params.get('win_reward', Game.params['win_reward'])
        self.kill_reward = params.get('win_reward', Game.params['kill_reward'])
        self.message_length = params.get(
            'message_length', Game.params['message_length'])
        self.impostor_frac = params.get(
            'impostor_frac', Game.params['impostor_frac'])
        self.vote_period = params.get(
            'vote_period', Game.params['vote_period'])

        self.reset()

    def reset(self):
        # Generate a new world
        self.world = World(self.n, self.t)

        # Pick num_agents different map locations
        rng = np.random.default_rng()
        locs = rng.choice(self.n * self.n, size=self.num_agents, replace=False)
        coord_map = np.vectorize(lambda x: (x//self.n, x % self.n))
        locs = np.vstack(coord_map(locs)).T

        # Instatntiate num_agents - 1 Crewmates and 1 Impostor agents
        self.impostor_index = random.randint(0, self.num_agents-1)
        self.impostor_indices = rng.choice(self.num_agents, size=max(
            int(self.num_agents * self.impostor_frac), 1), replace=False)
        self.impostor_set = set(self.impostor_indices)

        self.agents = []
        for i in range(self.num_agents):
            if i in self.impostor_set:
                self.agents.append(Impostor(i, self.world, locs[i], vision_radius=self.agent_vision_radius,
                                            message_length=self.message_length, kill_distance=self.kill_distance))
            else:
                self.agents.append(Crewmate(
                    i, self.world, locs[i], message_length=self.message_length, vision_radius=self.agent_vision_radius))
        self.world.set_agent_list(self.agents)

        # Whether the game is ongoing
        self.ongoing = True
        self.num_steps = 0

    def step(self, actions):
        # This function returns the new observation for each agent,
        # The reward they got,
        # Whether the game is done
        # to_kill = None
        rewards = np.zeros(self.num_agents)
        votes = []

        for j, action in enumerate(actions):
            # for agent in self.agents:
            # action = agent.get_action()

            # Assume action to be a row vector
            # Index 0 represents kill(0) or move(1)
            # Index 1 represents which direction to move
            # Index 2 represents kick vote
            # Remainder of the k*(n-1) indices represent messages to each other agent

            # Record Vote
            votes.append(action[2])
            agent = self.agents[j]

            # Dead agents cannot do anything
            if not agent.alive:
                continue

            if action[0] == 0:
                # Only the impostor can kill someone else
                if agent.id not in self.impostor_set:
                    continue

                # Choose closest alive agent within killing distance to kill
                visible_agents = agent.sense_agents()
                for va in visible_agents:
                    if va.alive and agent.distance_to(va.location) <= self.kill_distance:
                        # Kill Agent in range
                        # print('Agent {} killed Agent {}'.format(agent.id, va.id))
                        self.world.kill_agent(va.id)
                        rewards[va.id] -= self.kill_reward
                        rewards[agent.id] += self.kill_reward
                        break

            elif action[0] == 1:
                # We move, we move
                task_done = self.world.move_agent(agent.id, action[1])
                if task_done:
                    rewards[agent.id] = self.task_reward

        # # Kill someone
        # if to_kill is not None:
        #     self.world.kill_agent(to_kill.id)
        #     dead_agent = self.agents[to_kill.id]

        #     rewards[dead_agent.id] -= self.kill_reward
        #     for i in self.impostor_set:
        #         rewards[i] += self.kill_reward

        # Perform voting
        if (self.num_steps + 1) % self.vote_period == 0:
            votes = [v for j, v in enumerate(votes) if self.agents[j].alive]
            to_kick = max(set(votes), key=votes.count)
            to_kick = int(to_kick)

            if 0 <= to_kick < len(self.agents) and self.agents[to_kick].alive:
                self.world.kill_agent(to_kick)

       

        # Check kill win condition for impostor
        num_impostors = sum([self.agents[j].alive for j in self.impostor_set])
        if self.ongoing and num_impostors * 2 >= self.world.num_agents:
            # add_rewards = -self.win_reward * np.ones(self.num_agents) # Or 0?
            add_rewards = np.zeros(self.num_agents)

            for j in self.impostor_set:
              add_rewards[j] = self.win_reward

            rewards += add_rewards
            self.ongoing = False
            self.win_condition = 'Impostor Majority'

        # Check kill win condition for crewmates
        elif num_impostors <= 0:
            # add_rewards = self.win_reward * np.ones(self.num_agents)
            add_rewards = np.zeros(self.num_agents)

            for j in range(self.num_agents):
              if j not in self.impostor_set:
                add_rewards += self.win_reward

            # I think this does what I want it to...
            rewards = rewards + add_rewards
            self.ongoing = False
            self.win_condition = 'Impostors Erradicated'
        
        # Check task win condition
        if self.world.tasks_left <= 0:
          self.ongoing = False
          add_rewards = np.zeros(self.num_agents)

          for j in range(self.num_agents):
            if j not in self.impostor_set:
              add_rewards += self.win_reward
          rewards = rewards + add_rewards

          self.win_condition = 'Tasks Completed'

        self.num_steps += 1
        if self.num_steps >= self.max_steps:
          add_rewards = np.zeros(self.num_agents)
          for j in self.impostor_set:
            add_rewards[j] = self.win_reward

          rewards += add_rewards
          self.ongoing = False
          self.win_condition = 'Uncompleted Tasks'


        observations = self.get_observations()

        return (observations, rewards, self.ongoing)

    def get_observations(self):
        return [a.get_observation() for a in self.agents]

    def get_actions(self, model, obs):
        actions = []
        for a, o in zip(self.agents, obs):
            actions.append(model.get_action(a, o))
        return actions

    def __repr__(self) -> str:
        world_arr = np.array(np.maximum(
            self.world.map, self.world.agent_map), dtype=object)

        def make_blank(x):
            if x == 0.0:
                return ''
            elif x < 100.0:
                return 'G-{:^d}'.format(int(x))  # Goal Location
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
