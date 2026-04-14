
import isaacgym
from mqe.envs.utils import make_mqe_env
from mqe.utils import get_args
import torch
import numpy as np
import imageio

def visualize():
    args = get_args()
    args.task = "go1pushball"
    args.num_envs = 1
    args.headless = True # We will capture frames manually if possible, or just print coords
    args.record_video = False # We'll try to capture frames manually
    
    # Force CPU for easier debugging if needed, but GPU is fine
    
    env, cfg = make_mqe_env(args.task, args)
    env.reset()
    
    frames = []
    
    print(f"Env Origins: {env.env_origins}")
    print(f"Agent Origins: {env.agent_origins}")
    
    for i in range(20):
        # No action
        obs, reward, done, info = env.step(torch.zeros(1, 2, 3, device=env.device))
        
        # Capture frame if possible (requires headless=False usually, or virtual display)
        # Since we are headless, we might not get images easily without a virtual display.
        # Let's rely on coordinate printing first.
        
        root_states = env.root_states
        ball_pos = env.root_states_npc[:, :3]
        
        print(f"Step {i}")
        print(f"  Robot 0 Pos (Global): {root_states[0, :3].cpu().numpy()}")
        print(f"  Robot 1 Pos (Global): {root_states[1, :3].cpu().numpy()}")
        print(f"  Ball Pos (Global): {ball_pos[0].cpu().numpy()}")
        
        # Check relative to env origin
        env_origin = env.env_origins[0].cpu().numpy()
        print(f"  Robot 0 Rel: {root_states[0, :3].cpu().numpy() - env_origin}")
        print(f"  Ball Rel: {ball_pos[0].cpu().numpy() - env_origin}")

if __name__ == "__main__":
    visualize()
