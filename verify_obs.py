
import isaacgym
from mqe.envs.utils import make_mqe_env
from mqe.utils import get_args
import torch
import numpy as np

def verify_obs():
    args = get_args()
    args.task = "go1pushball"
    args.num_envs = 1
    args.headless = True
    
    env, cfg = make_mqe_env(args.task, args)
    env.reset()
    
    # Run a few steps
    print(f"Env Origins: {env.env_origins}")
    for i in range(5):
        obs, reward, done, info = env.step(torch.zeros(1, 2, 3, device=env.device))
        
        # Obs shape: (num_envs, num_agents, 22)
        # 0-1: ID
        # 2-7: Self State (pos, rot)
        # 8-13: Partner State
        # 14-15: Hole Pos
        # 16-17: Ball Pos
        # 18-21: Ball Rot
        
        print(f"--- Step {i} ---")
        agent0_obs = obs[0, 0, :].cpu().numpy()
        
        self_pos = agent0_obs[2:5]
        hole_pos = agent0_obs[14:16]
        ball_pos = agent0_obs[16:18]
        
        print(f"Agent 0 Self Pos: {self_pos}")
        print(f"Hole Pos (Rel): {hole_pos}")
        print(f"Ball Pos (Rel): {ball_pos}")
        
        # Verify logic
        # Ball should be around (2.0, 0.0)
        # Hole should be around (3.5, 0.0) (Block length 2.0 + 1.5?)
        # Let's check barrier_track.py for hole position logic
        
verify_obs()
