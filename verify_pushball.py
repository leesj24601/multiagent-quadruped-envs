
import numpy as np
import os
import isaacgym
from mqe.utils import get_args
import torch
from mqe.envs.utils import make_mqe_env, custom_cfg

if __name__ == '__main__':
    args = get_args()
    task_name = "go1pushball"
    args.num_envs = 1
    args.headless = True # Run headless for verification
    args.record_video = False

    env, env_cfg = make_mqe_env(task_name, args, custom_cfg(args))
    env.reset()
    
    print("Environment created and reset successfully.")
    
    for i in range(100):
        # Action: move forward to push the ball
        # Action shape: (num_envs, num_agents, 3)
        # We have 2 agents in config?
        # Check num_agents
        num_agents = env.num_agents
        action = torch.tensor([[[1, 0, 0]] * num_agents], dtype=torch.float32, device="cuda").repeat(env.num_envs, 1, 1)
        
        obs, reward, done, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i}, Reward: {reward.mean().item()}")
            # Check if hole_pos is in obs
            # The wrapper adds hole_pos at index ...
            # obs shape is (num_envs, num_agents, 21 + num_agents)
            # We can check env.hole_pos directly if we want
            if hasattr(env, 'hole_pos'):
                 print(f"Hole Pos (relative): {env.hole_pos[0, 0]}")

    print("Verification finished.")
