import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg
from mqe.envs.configs.go1_pushbox_config import Go1PushboxCfg

class Go1PushboxMediumCfg(Go1PushboxCfg):

    class env(Go1PushboxCfg.env):
        env_name = "go1pushbox_medium"
        num_agents = 2
    
    class asset(Go1PushboxCfg.asset):
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/box_medium.urdf"
        name_npc = "box"

    class rewards(Go1PushboxCfg.rewards):
        class scales(Go1PushboxCfg.rewards.scales):
            box_x_movement_reward_scale = 100
