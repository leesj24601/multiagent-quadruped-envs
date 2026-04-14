
import isaacgym
from openrl_ws.utils import make_env, get_args, custom_cfg
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent
import torch
import numpy as np

def verify_physics():
    args = get_args()
    args.headless = False # Force visualization
    args.num_envs = 1
    
    # Create environment
    env, _ = make_env(args, custom_cfg(args))
    
    # Reset
    obs = env.reset()
    
    print("Starting physics verification...")
    print("Forcing robot to move forward (action=[1.0, 0.0, 0.0])...")
    
    for i in range(500):
        # Force forward action
        # Action space is 3: [vx, vy, wz]
        # We want max forward velocity
        action = torch.tensor([[1.0, 0.0, 0.0]], device="cuda").repeat(env.num_envs, env.num_agents, 1).cpu().numpy()
        
        # Step
        obs, r, done, info = env.step(action)
        
        # Get ball position
        # Assuming single env, single ball
        # Accessing internal state for debugging
        # env.env is the wrapper, env.env.env is the Go1PushBall env
        if hasattr(env, "env") and hasattr(env.env, "root_states_npc"):
             ball_pos = env.env.root_states_npc[0, :3]
             print(f"Step {i}: Ball Pos: {ball_pos.cpu().numpy()}")
        
        if done.any():
            obs = env.reset()

    print("Verification finished.")

if __name__ == '__main__':
    verify_physics()
