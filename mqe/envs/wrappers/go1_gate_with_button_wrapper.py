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
        
        # Store initial gate position (assuming 1 NPC which is the gate)
        # We need to wait for reset to get the actual tensor values, but we can prepare indices
        self.gate_indices = None

        self.reward_buffer = {
            "button press reward": 0,
            "success reward": 0,
            "step count": 0
        }

    def reset(self):
        obs_buf = self.env.reset()
        
        # Initialize gate indices if not done
        if self.gate_indices is None:
            # Assuming the last actors are NPCs. 
            # In Go1Cfg, num_agents=2, num_npcs=1.
            # Actor indices in root_states: [agent1, agent2, ..., npc1, ...]
            # We need to find the global index of the gate NPC.
            # The environment should have `actor_indices` or similar.
            # In LeggedRobot, `self.actor_indices` contains indices for all actors including envs.
            # But we want to modify specific actors.
            # Let's rely on `self.env.root_states_npc` which is a view or copy?
            # Actually `LeggedRobot` updates `self.root_states` and then `self.gym.set_actor_root_state_tensor`.
            # We need to modify `self.env.root_states` directly at the NPC index.
            pass

        # Initial observation construction (similar to pushball)
        base_pos = obs_buf.base_pos.view(self.env.num_envs, self.env.num_agents, 3)
        base_pos = base_pos.view(-1, 3)
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        
        # Add button position to obs
        # Button pos is static relative to env origin? 
        # Yes, defined in cfg.game.button_pos.
        # We should broadcast it.
        button_pos_batch = self.button_pos.unsqueeze(0).unsqueeze(0).repeat(self.env.num_envs, self.num_agents, 1) # (num_envs, num_agents, 2)
        
        # Gate pos (NPC)
        gate_pos = self.env.root_states_npc[:, :3] - self.env.env_origins
        gate_pos_batch = gate_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1)

        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         button_pos_batch, gate_pos_batch,
                         self.root_states_npc[:, 3:7].unsqueeze(1).repeat(1, self.num_agents, 1)], dim=2)

        return obs

    def step(self, action):
        # 1. Check Button Logic BEFORE physics step (to open gate for this step)
        # Get agent positions relative to env origin
        # self.env.base_pos is (num_envs * num_agents, 3)
        # self.env.env_origins is (num_envs, 3)
        # We need to handle the broadcasting carefully.
        
        # Reshape base_pos to (num_envs, num_agents, 3)
        current_base_pos = self.env.base_pos.view(self.env.num_envs, self.env.num_agents, 3)
        # env_origins = self.env.env_origins.unsqueeze(1) # (num_envs, 1, 3)
        # rel_pos = current_base_pos - env_origins
        
        # But wait, self.env.base_pos is absolute world coordinates.
        # self.button_pos is relative to env origin (from config).
        # So we need to compare (base_pos - env_origin) with button_pos.
        
        # Calculate distance to button
        # button_pos is (2,)
        # rel_pos[:, :, :2] is (num_envs, num_agents, 2)
        
        rel_pos = current_base_pos - self.env.env_origins.unsqueeze(1)
        dists_to_button = torch.norm(rel_pos[:, :, :2] - self.button_pos, dim=2) # (num_envs, num_agents)
        
        # Check if ANY agent is on the button
        button_pressed = torch.any(dists_to_button < self.button_radius, dim=1) # (num_envs,)
        
        # DEBUG: Print distance and pressed state
        if self.env.num_envs == 1:
            print(f"DEBUG: Base Pos: {current_base_pos[0, 0].tolist()}")
            print(f"DEBUG: Env Origin: {self.env.env_origins[0].tolist()}")
            print(f"DEBUG: Rel Pos: {rel_pos[0, 0].tolist()}")
            print(f"DEBUG: Button Pos: {self.button_pos.tolist()}")
            print(f"DEBUG: Dist: {dists_to_button[0].tolist()}, Pressed: {button_pressed[0].item()}")
        
        # 2. Move Gate
        # We need to modify self.env.root_states for the NPCs.
        # The NPC indices in root_states.
        # Assuming 1 NPC per env.
        # self.env.root_states is (num_envs * (num_agents + num_npcs), 13)
        # The layout is usually [agent1_env0, agent2_env0, npc1_env0, agent1_env1, ... ] OR grouped by type?
        # Isaac Gym standard: actors are added sequentially per env.
        # So for each env: Agent1, Agent2, NPC1.
        # Index of NPC in env i: i * (num_agents + num_npcs) + num_agents
        
        total_actors_per_env = self.num_agents + self.env.num_npcs
        npc_indices = torch.arange(self.env.num_envs, device=self.device) * total_actors_per_env + self.num_agents
        
        # Update Z position of gates
        # We need to set the state in the simulation.
        # self.env.root_states is a buffer. We update it and then call set_actor_root_state_tensor_indexed.
        
        # Get current NPC states
        # We can just modify the z-coordinate.
        
        # Default closed height (from config or current)
        # We want to set it to open_height if pressed, closed_height if not.
        
        target_z = torch.where(button_pressed, self.gate_open_height, self.gate_closed_height)
        
        # We need to be careful not to reset X, Y, Rot if they moved (though they are fixed base, so they shouldn't move).
        # But to be safe, we just update Z.
        # However, root_states contains global coordinates.
        # So we need to keep X, Y as they are.
        
        # We can access self.env.root_states directly?
        # Yes, LeggedRobot has self.root_states.
        
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

        # Button Reward (Shared)
        # Give reward if button is pressed
        button_reward_scale = 0.1 # Decreased from 1.0 to avoid local optimum
        reward += button_pressed.unsqueeze(1).float() * button_reward_scale
        self.reward_buffer["button press reward"] += torch.sum(button_pressed.float()).cpu()

        # Distance Penalty (Encourage separation)
        # Calculate distance between agents
        # rel_pos is (num_envs, num_agents, 3)
        # Assuming 2 agents
        if self.num_agents == 2:
            dist_between_agents = torch.norm(rel_pos[:, 0, :2] - rel_pos[:, 1, :2], dim=1)
            # Penalty if too close (e.g., < 1.0m)
            too_close = dist_between_agents < 1.5
            reward[too_close, :] -= 0.1 # Penalty for both agents

        # Success Reward (Passing the gate)
        # Check if agents are past the gate.
        # Gate is at x = ~3.5 (relative).
        # Let's say target is x > 4.0.
        
        # success = torch.all(rel_pos[:, :, 0] > 4.0, dim=1) # Both agents passed?
        # Or just one? Task description said "cooperate to pass".
        # Let's reward if ANY agent passes the gate line.
        # But we want to avoid them just spawning past it (init is at 0).
        
        agents_past_gate = rel_pos[:, :, 0] > 4.0
        success = torch.any(agents_past_gate, dim=1) # At least one passed
        
        reward[success, :] += self.env.cfg.rewards.scales.success_reward_scale
        
        # Terminate if success
        termination[success] = True
        self.reward_buffer["success reward"] += torch.sum(success.float()).cpu()

        # Construct Observation (same as reset)
        base_pos = obs_buf.base_pos.view(self.env.num_envs, self.env.num_agents, 3)
        base_pos = base_pos.view(-1, 3)
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        
        button_pos_batch = self.button_pos.unsqueeze(0).unsqueeze(0).repeat(self.env.num_envs, self.num_agents, 1)
        gate_pos = self.env.root_states_npc[:, :3] - self.env.env_origins
        gate_pos_batch = gate_pos[:, :2].unsqueeze(1).repeat(1, self.num_agents, 1)

        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1]),
                         button_pos_batch, gate_pos_batch,
                         self.root_states_npc[:, 3:7].unsqueeze(1).repeat(1, self.num_agents, 1)], dim=2)

        return obs, reward, termination, info
