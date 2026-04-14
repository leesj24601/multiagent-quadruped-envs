
import isaacgym
from mqe.envs import *
from mqe.utils import task_registry
import torch

def verify_environment():
    env_cfg, train_cfg = task_registry.get_cfgs(name="go1gatewithbutton")
    
    # Override some settings for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    
    # Prepare environment
    env, _ = task_registry.make_env(name="go1gatewithbutton", args=None, env_cfg=env_cfg)
    
    print("Environment created successfully.")
    
    obs = env.reset()
    print("Reset successful. Observation shape:", obs.shape)
    
    # Run a few steps
    for i in range(10):
        # Random actions
        actions = torch.rand((env.num_envs, env.num_agents, 3), device=env.device) * 2 - 1
        obs, rewards, dones, infos = env.step(actions)
        print(f"Step {i}: Reward shape {rewards.shape}, Done {dones}")
        
    print("Verification complete.")

if __name__ == "__main__":
    verify_environment()
