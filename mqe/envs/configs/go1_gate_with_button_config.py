import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1GateWithButtonCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1gatewithbutton"
        num_envs = 1
        num_agents = 2
        episode_length_s = 15 # episode length in seconds
        num_npcs = 1 # Gate object

    class asset(Go1Cfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/gate.urdf" # Using gate URDF
        name_npc = "gate"
        npc_collision = True
        fix_npc_base_link = True # Initially fixed, moved by wrapper
        npc_gravity = False # No gravity needed if we control position manually

    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "gate", # The wall with gap
                "plane",
                "wall",
            ],
            track_width = 3.0,
            init = dict(
                block_length = 2.0,
                room_size = (1.0, 1.5),
                border_width = 0.00,
                offset = (0, 0),
            ),
            gate = dict(
                block_length = 3.0,
                width = 0.6, # Gap width
                depth = 0.1, 
                offset = (0, 0),
                random = (0.0, 0.0), # Fixed position for easier cooperation logic first
            ),
            plane = dict(
                block_length = 2.0,
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False,
            no_perlin_threshold = 0.06,
            add_perlin_noise = False,
       ))
        
    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [0.0, 1.0, 0.42], # Moved to 1.0
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [0.0, -1.0, 0.42], # Moved to -1.0
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        # Gate initial position (blocking the gap)
        # Gap is at x = 2.0 + 3.0/2 = 3.5 roughly? 
        # init block 2.0. gate block starts at 2.0.
        # Gate object should be placed where the gap is.
        # Based on go1_gate_config, the gate terrain creates a wall with a hole.
        # We need a movable object to block that hole.
        # Actually, 'gate' in terrain creates the static walls.
        # We need to place the NPC (movable gate) in the gap.
        # Let's assume the gap is at local x=1.5 within the gate block.
        # Total x from start = 2.0 (init) + 1.5 = 3.5.
        init_states_npc = [
            init_state_class(
                pos = [3.5, 0.0, 0.5], # Adjust based on actual terrain generation
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class control(Go1Cfg.control):
        control_type = 'C'

    class termination(Go1Cfg.termination):
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )
        init_npc_base_pos_range = None
 
    class rewards(Go1Cfg.rewards):
        class scales:
            target_reward_scale = 1
            success_reward_scale = 50 # Increased from 10 to encourage passing the gate
            # Button reward will be handled in wrapper
            contact_punishment_scale = -1
            
    class viewer(Go1Cfg.viewer):
        pos = [-2., 2.5, 4.]  # [m]
        lookat = [4., 2.5, 0.]  # [m]

    class game:
        # Custom config for the game logic
        button_pos = [3.0, -1.0] # In front of the right wall (Y < 0)
        button_radius = 0.5
        gate_open_height = 2.0
        gate_closed_height = 0.5
