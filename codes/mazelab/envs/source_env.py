from mazelab import BaseEnv
from mazelab import BaseMotion

import gym
from gym.spaces import Box
from gym.spaces import Discrete
from .source_maze import SourceMaze
from.target_maze import TargetMaze
import numpy as np
from copy import deepcopy


class SourceEnv(BaseEnv):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        if not self.invert:
            self.maze = SourceMaze(**kwargs)
        else:
            self.maze = TargetMaze(**kwargs)

        self.motions = BaseMotion()
        self.observation_space = Box(low = 0, high = len(self.maze.objects), shape = self.maze.size, dtype = np.uint8)
        self.action_space = Discrete(len(self.motions))
        self.switches = len(self.maze.objects.switch.positions)
        self.prize_count = 0

    def step(self, action):
        motion = self.motions[action]
        curr_position = self.maze.objects.agent.positions[0]
        new_position = [curr_position[0] + motion[0], curr_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        done = False
        reward = 0

        if self._is_switch(curr_position, new_position):
            self.switches = self.switches - 1

        if self._is_prize(curr_position, new_position):
            if self._is_activated():
                reward = 1
                self.prize_count = self.prize_count + 1
                if self.prize_count == len(self.initial_positions["prize"]):
                    # self.logger.info("Reward {} Prize count {}".format(reward, self.prize_count))
                    done = True
            else:
                reward = -1

        if self._is_obstacle(new_position):
            reward = 0

        if valid:
            self.maze.objects.agent.positions = [new_position]
            a = self.maze.objects.free.positions
            self.maze.objects.free.positions = [ x for x in a if not (x[0] == new_position[0] and x[1] == new_position[1])]
            check =  [ x for x in a if (x[0] == curr_position[0] and x[1] == curr_position[1])]
            if len(check) == 0:
                self.maze.objects.free.positions.append(curr_position)

        if self.return_image:
            state = self.render(mode = 'rgb_array')
        else:
            state = self.maze.to_value()
        return state, reward, done , {}

    def reset(self):
        self.maze.objects.agent.positions = self.start_idx
        # self.maze.objects.free.positions = [ x for x in self.initial_positions["free"] if not (x[0] == self.start_idx[0][0] and x[1] == self.start_idx[0][1])]
        # # self.maze.objects.free.positions = self.initial_positions["free"]
        # self.maze.objects.switch.positions = self.initial_positions["switch"]
        # self.maze.objects.prize.positions = self.initial_positions["prize"]
        if self.return_image:
            return self.render(mode = 'rgb_array')
        else:
            return self.maze.to_value()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] <  self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_switch(self, curr_position, new_position):
        out = False
        for pos in self.maze.objects.switch.positions:
            if new_position[0] == pos[0] and new_position[1] == pos[1]:
                out = True
                self.maze.x[pos[0],pos[1]] = self.maze.objects.free.value
                self.maze.reset_maze(new_position)
                break
        return out

    def _is_prize(self, curr_position, new_position):
        out = False
        for pos in self.maze.objects.prize.positions:
            if new_position[0] == pos[0] and new_position[1] == pos[1]:
                out = True
                if self._is_activated():
                    self.maze.x[pos[0],pos[1]] = self.maze.objects.free.value
                    self.maze.reset_maze(new_position)
                break
        return out

    def _is_obstacle(self, position):
        out = False
        for pos in self.maze.objects.obstacle.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def _is_activated(self):
        return self.switches == 0

    def get_image(self):
        return self.maze.to_rgb()
