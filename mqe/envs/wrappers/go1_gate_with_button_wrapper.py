import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from isaacgym import gymtorch, gymapi

class Go1GateWithButtonWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(20 + self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        # Game logic parameters
        self.button_pos = torch.tensor(self.env.cfg.game.button_pos, device=self.device)
        self.button_radius = self.env.cfg.game.button_radius
        self.gate_open_height = self.env.cfg.game.gate_open_height
        self.gate_closed_height = self.env.cfg.game.gate_closed_height
        self.success_x = getattr(self.env.cfg.game, "success_x", 4.0)
        self.button_reward_scale = getattr(self.env.cfg.game, "button_reward_scale", 0.1)
        self.button_approach_reward_scale = getattr(self.env.cfg.game, "button_approach_reward_scale", self.target_reward_scale)
        self.forward_progress_reward_scale = getattr(self.env.cfg.game, "forward_progress_reward_scale", self.target_reward_scale)
        self.min_agent_distance = getattr(self.env.cfg.game, "min_agent_distance", 1.5)
        self.separation_penalty_scale = getattr(self.env.cfg.game, "separation_penalty_scale", -0.1)

        self.last_button_dists = None

        self.reward_buffer = {
            "button approach reward": 0,
            "forward progress reward": 0,
            "button press reward": 0,
            "separation punishment": 0,
            "success reward": 0,
            "step count": 0
        }

    def _relative_base_pos(self):
        base_pos = self.env.base_pos.view(self.env.num_envs, self.env.num_agents, 3)
        return base_pos - self.env.env_origins.unsqueeze(1)

    def _button_dists(self, rel_pos):
        return torch.norm(rel_pos[:, :, :2] - self.button_pos, dim=2)

    def _build_obs(self, obs_buf):
        base_pos = obs_buf.base_pos.view(self.env.num_envs, self.env.num_agents, 3)
        base_pos = base_pos.view(-1, 3)
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        button_pos_batch = self.button_pos.unsqueeze(0).unsqueeze(0).repeat(self.env.num_envs, self.num_agents, 1)
        gate_pos = self.env.root_states_npc[:, :3] - self.env.env_origins
        gate_pos_batch = gate_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1)

        return torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                          button_pos_batch, gate_pos_batch,
                          self.root_states_npc[:, 3:7].unsqueeze(1).repeat(1, self.num_agents, 1)], dim=2)

    def reset(self):
        obs_buf = self.env.reset()
        self.last_button_dists = self._button_dists(self._relative_base_pos())
        return self._build_obs(obs_buf)

    def step(self, action):
        rel_pos_before = self._relative_base_pos()
        dists_to_button_before = self._button_dists(rel_pos_before)
        button_pressed = torch.any(dists_to_button_before < self.button_radius, dim=1)

        npc_indices = self.env.npc_indices[:, 0].long()
        target_z = torch.where(button_pressed, self.gate_open_height, self.gate_closed_height)
        self.env.all_root_states[npc_indices, 2] = target_z + self.env.env_origins[:, 2]

        self.env.gym.set_actor_root_state_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.env.all_root_states),
            gymtorch.unwrap_tensor(npc_indices.int()),
            len(npc_indices)
        )

        # 3. Step Environment
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))
        rel_pos_after = obs_buf.base_pos.view(self.env.num_envs, self.env.num_agents, 3)
        dists_to_button_after = self._button_dists(rel_pos_after)

        # Debug Visualization: Draw Button
        # Always try to draw if num_envs is small enough
        if self.env.num_envs <= 5: 
            # print("DEBUG: Drawing button lines...") # Uncomment if needed
            self.env.gym.clear_lines(self.env.viewer)
            
            for i in range(self.env.num_envs):
                # Draw a circle (approximated by lines) or cross for the button
                # Button pos is relative to env origin.
                # Since add_lines draws in global coords (because env handles are at 0,0,0),
                # we must add env_origins to button_pos to draw it correctly.
                
                # Get env origin for this env
                origin = self.env.env_origins[i]
                
                button_x = origin[0].item() + self.button_pos[0].item()
                button_y = origin[1].item() + self.button_pos[1].item()
                z = origin[2].item() + 0.05 # Slightly above ground
                r = self.button_radius
                
                # Check if pressed for this env
                is_pressed = button_pressed[i].item()
                line_color = [1.0, 0.0, 0.0] if not is_pressed else [0.0, 1.0, 0.0]
                
                # Circle approximation
                num_segments = 20
                angle_step = 2 * numpy.pi / num_segments
                lines = []
                for j in range(num_segments):
                    angle1 = j * angle_step
                    angle2 = (j + 1) * angle_step
                    x1, y1 = button_x + r * numpy.cos(angle1), button_y + r * numpy.sin(angle1)
                    x2, y2 = button_x + r * numpy.cos(angle2), button_y + r * numpy.sin(angle2)
                    lines.append([x1, y1, z, x2, y2, z])
                
                # Add a vertical marker at center
                lines.append([button_x, button_y, z, button_x, button_y, z + 0.5])

                verts = []
                for l in lines:
                    verts.extend(l)
                
                self.env.gym.add_lines(self.env.viewer, self.env.envs[i], len(lines), verts, line_color)

        # 4. Calculate Rewards
        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)

        active_envs = ~termination.to(torch.bool)

        if self.last_button_dists is None:
            self.last_button_dists = dists_to_button_before

        closest_button_before = self.last_button_dists.min(dim=1).values
        closest_button_after = dists_to_button_after.min(dim=1).values
        button_approach_reward = closest_button_before - closest_button_after
        button_approach_reward = torch.clamp(button_approach_reward, -0.2, 0.2)
        button_approach_reward = torch.where(
            active_envs,
            button_approach_reward * self.button_approach_reward_scale,
            torch.zeros_like(button_approach_reward),
        )
        reward += button_approach_reward.unsqueeze(1)
        self.reward_buffer["button approach reward"] += torch.sum(button_approach_reward).cpu()

        forward_progress_reward = rel_pos_after[:, :, 0] - rel_pos_before[:, :, 0]
        not_past_success = rel_pos_after[:, :, 0] < self.success_x
        forward_progress_reward = torch.clamp(forward_progress_reward, -0.2, 0.2)
        forward_progress_reward = torch.where(
            active_envs.unsqueeze(1) & not_past_success,
            forward_progress_reward * self.forward_progress_reward_scale,
            torch.zeros_like(forward_progress_reward),
        )
        reward += forward_progress_reward
        self.reward_buffer["forward progress reward"] += torch.sum(forward_progress_reward).cpu()

        # Button Reward (Shared)
        # Give reward if button is pressed
        button_press_reward = button_pressed.float() * self.button_reward_scale
        reward += button_press_reward.unsqueeze(1)
        self.reward_buffer["button press reward"] += torch.sum(button_press_reward).cpu()

        # Distance Penalty (Encourage separation)
        # Calculate distance between agents
        # rel_pos is (num_envs, num_agents, 3)
        # Assuming 2 agents
        if self.num_agents == 2:
            dist_between_agents = torch.norm(rel_pos_after[:, 0, :2] - rel_pos_after[:, 1, :2], dim=1)
            # Penalty if too close (e.g., < 1.0m)
            too_close = dist_between_agents < self.min_agent_distance
            separation_penalty = too_close.float() * self.separation_penalty_scale
            reward += separation_penalty.unsqueeze(1)
            self.reward_buffer["separation punishment"] += torch.sum(separation_penalty).cpu()

        # Success Reward (Passing the gate)
        # Check if agents are past the gate.
        # Gate is at x = ~3.5 (relative).
        # Let's say target is x > 4.0.
        
        # success = torch.all(rel_pos[:, :, 0] > 4.0, dim=1) # Both agents passed?
        # Or just one? Task description said "cooperate to pass".
        # Let's reward if ANY agent passes the gate line.
        # But we want to avoid them just spawning past it (init is at 0).
        
        agents_past_gate = rel_pos_after[:, :, 0] > self.success_x
        success = torch.any(agents_past_gate, dim=1) # At least one passed
        
        success_reward = success.float() * self.success_reward_scale
        reward += success_reward.unsqueeze(1)
        
        # Terminate if success
        termination[success] = True
        self.reward_buffer["success reward"] += torch.sum(success_reward).cpu()
        self.last_button_dists = dists_to_button_after

        # Construct Observation (same as reset)
        return self._build_obs(obs_buf), reward, termination, info
