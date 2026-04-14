import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym.torch_utils import *

class Go1PushBallWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(20 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        # for hard setting of reward scales (not recommended)
        
        self.ball_x_movement_reward_scale = 1
        self.hole_reward_scale = 500.0

        self.reward_buffer = {
            "ball movement reward": 0,
            "approach ball reward": 0,
            "hole reward": 0,
            "step count": 0
        }

    def _init_extras(self, obs):
        # Access hole_pos from env.env_info (already broadcasted to num_envs)
        if hasattr(self.env, "env_info") and "hole_pos" in self.env.env_info:
             self.hole_pos = self.env.env_info["hole_pos"]
        elif hasattr(self.env, "terrain") and hasattr(self.env.terrain, "env_info"):
             # Fallback if env_info not present in env but in terrain (e.g. raw grid)
             # We need to broadcast it manually using terrain_levels if available
             hole_pos_grid = self.env.terrain.env_info["hole_pos"]
             if hasattr(self.env, "terrain_levels") and hasattr(self.env, "terrain_types"):
                 self.hole_pos = hole_pos_grid[self.env.terrain_levels, self.env.terrain_types]
             else:
                 # Last resort: assume 1x1 grid and repeat
                 self.hole_pos = hole_pos_grid.view(1, -1).repeat(self.env.num_envs, 1)
        else:
             # Fallback if accessed differently (e.g. via extras)
             self.hole_pos = obs.env_info["hole_pos"]

        # Make relative to env origin
        self.hole_pos = self.hole_pos - self.env.env_origins
        self.hole_pos = self.hole_pos.unsqueeze(1).repeat(1, self.num_agents, 1)

    def reset(self):
        obs_buf = self.env.reset()

        if getattr(self, "hole_pos", None) is None:
            self._init_extras(obs_buf)

        ball_pos = self.root_states_npc[:, :3] - self.env.env_origins
        
        base_pos = obs_buf.base_pos.view(self.env.num_envs, self.env.num_agents, 3)
        # base_pos is already relative to env_origins in obs_buf
        base_pos = base_pos.view(-1, 3)

        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         self.hole_pos[:, :, :2], ball_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1),
                         self.root_states_npc[:, 3:7].unsqueeze(1).repeat(1, self.num_agents, 1)], dim=2)

        self.last_ball_pos = None
        self.last_dists_to_ball = None

        return obs

    def step(self, action):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        ball_pos = self.root_states_npc[:, :3] - self.env.env_origins
        
        if getattr(self, "hole_pos", None) is None:
            self._init_extras(obs_buf)
        
        base_pos = obs_buf.base_pos.view(self.env.num_envs, self.env.num_agents, 3)
        # base_pos is already relative to env_origins in obs_buf
        base_pos = base_pos.view(-1, 3)

        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         self.hole_pos[:, :, :2], ball_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1),
                         self.root_states_npc[:, 3:7].unsqueeze(1).repeat(1, self.num_agents, 1)], dim=2)

        self.reward_buffer["step count"] += 1
        # Initialize reward as (num_envs, num_agents) to handle individual rewards
        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)

        # 1. Ball Distance Improvement Reward (Shared)
        # Reward = (Previous Distance - Current Distance) * Scale
        # If ball gets closer, reward is positive. If it moves away, reward is negative.
        if self.last_ball_pos is not None:
            target_pos = self.hole_pos[:, 0, :2] # (num_envs, 2)
            
            # Current distance
            curr_dist = torch.norm(ball_pos[:, :2] - target_pos, dim=1)
            
            # Previous distance
            prev_dist = torch.norm(self.last_ball_pos[:, :2] - target_pos, dim=1)
            
            # Improvement
            dist_improvement = prev_dist - curr_dist
            
            scale = getattr(self, "ball_x_movement_reward_scale", 100.0)
            # Broadcast shared reward to all agents
            reward += dist_improvement.unsqueeze(1) * scale
            
            self.reward_buffer["ball movement reward"] += torch.sum(dist_improvement).cpu()

        # 2. Approach Ball Reward (Individual)
        # Encourage robots to get closer to the ball
        
        # Reshape base_pos back to (num_envs, num_agents, 3) for distance calculation
        base_pos_3d = base_pos.view(self.env.num_envs, self.env.num_agents, 3)
        
        if self.last_dists_to_ball is not None:
            # Calculate current distance to ball for each agent
            # base_pos_3d: (num_envs, num_agents, 3)
            # ball_pos: (num_envs, 3)
            dists_to_ball = torch.norm(base_pos_3d[:, :, :2] - ball_pos[:, :2].unsqueeze(1), dim=2) # (num_envs, num_agents)
            
            # Improvement
            approach_improvement = self.last_dists_to_ball - dists_to_ball
            
            # Scale: 5.0 (Smaller than ball movement, but enough to guide)
            approach_scale = 5.0
            # Add individual reward directly (shapes match: [num_envs, num_agents])
            reward += approach_improvement * approach_scale
            
            self.reward_buffer["approach ball reward"] += torch.sum(approach_improvement).cpu()
            
        self.last_dists_to_ball = torch.norm(base_pos_3d[:, :, :2] - ball_pos[:, :2].unsqueeze(1), dim=2)

        # 3. Contact Reward (Individual)
        # Encourage touching the ball
        # Ball radius 0.5, Robot radius ~0.3. Threshold 0.8.
        is_contacting = self.last_dists_to_ball < 0.8
        contact_reward_scale = 0.1
        reward += is_contacting.float() * contact_reward_scale

        # Success reward (Goal in) (Shared)
        target_pos = self.hole_pos[:, 0, :]
        ball_dist = torch.norm(ball_pos - target_pos, dim=1)
        success = ball_dist < 0.2
        # Broadcast success reward to all agents
        reward[success, :] += self.hole_reward_scale
        
        # Terminate episode on success
        termination[success] = True
        
        # Shared reward for all agents -> No longer needed as reward is already (num_envs, num_agents)
        # reward = reward.repeat(1, self.num_agents)

        self.last_ball_pos = copy(ball_pos)

        return obs, reward, termination, info
