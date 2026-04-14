import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1PushBallCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1pushball"
        num_envs = 1
        num_agents = 2
        num_npcs = 1
        episode_length_s = 15
    
    class asset(Go1Cfg.asset):
        terminate_after_contacts_on = []
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/ball_heavy.urdf"
        name_npc = "ball_heavy"
        npc_collision = True
        fix_npc_base_link = False
        npc_gravity = True
    
    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "hole_wall",
            ],
            # wall_thickness= 0.2,
            track_width = 5.0,
            init = dict(
                block_length = 2.0,
                room_size = (1.0, 2.5),
                border_width = 0.0,
                offset = (0, 0),
            ),
            hole_wall = dict(
                block_length = 3.0,
                hole_radius = 0.6,
                hole_center_z = 0.6,
            ),
            wall_height= 1.5,
            virtual_terrain = False, # Change this to False for real terrain
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
                pos = [0.0, 0.15, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [0.0, -0.15, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            init_state_class(
                pos = [2.0, 0.0, 0.6],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class control(Go1Cfg.control):
        control_type = 'C'

    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        # push_robots = True # use for virtual training
        push_robots = False # use for non-virtual training
        init_base_pos_range = dict(
            x= [-0.1, 0.1],
            y= [-0.1, 0.1],
        )
        init_npc_base_pos_range = dict(
            x= [-0.2, 0.2],
            y= [-0.2, 0.2],
        )

    class rewards(Go1Cfg.rewards):
        class scales:
            ball_x_movement_reward_scale = 100
            # Disable default rewards to force focus on the ball
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            # Re-enable penalties to prevent cheating
            torques = -0.0002
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -0.0 
            feet_air_time =  0.0
            collision = -1.0
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.0

    class viewer(Go1Cfg.viewer):
        pos = [0., 6., 5.]  # [m]
        lookat = [4., 6., 0.]  # [m]
