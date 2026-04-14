
import numpy as np
import os

import imageio
import isaacgym
from mqe.utils import get_args
import torch

from mqe.envs.utils import make_mqe_env, custom_cfg

def save_gif(frames, fps):
    print(f"Frames shape before saving: {frames.shape}")
    # frames.shape should be (N, H, W, C) for imageio
    
    # Define the output GIF file name
    output_gif_path = 'output_animation.gif'

    # Convert the frames to uint8 (assuming it's in range 0-1 or 0-255)
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)

    # imageio expects list of arrays or 4D array (N, H, W, C)
    # If it's (N, C, H, W), we need to transpose.
    if frames.shape[1] in [3, 4] and frames.shape[2] > 10 and frames.shape[3] > 10:
         # Likely NCHW
         print("Transposing NCHW to NHWC")
         frames = np.transpose(frames, (0, 2, 3, 1))

    imageio.mimsave(output_gif_path, frames, fps=fps)

    print("GIF created successfully.")

if __name__ == '__main__':
    args = get_args()

    # task_name = "go1plane"
    # task_name = "go1plane"
    # task_name = "go1gate"
    # task_name = "go1football-defender"
    # task_name = "go1football-1vs1"
    # task_name = "go1football-2vs2"
    # task_name = "go1sheep-easy"
    # task_name = "go1sheep-hard"
    # task_name = "go1seesaw"
    # task_name = "go1door"
    # task_name = "go1pushbox"
    # task_name = "go1pushbox"
    task_name = "go1pushball"
    # task_name = "go1tug"
    # task_name = "go1wrestling"
    # task_name = "go1rotationdoor"
    # task_name = "go1bridge"

    args.num_envs = 1
    args.headless = False
    args.record_video = False

    env, env_cfg = make_mqe_env(task_name, args, custom_cfg(args))
    if args.record_video:
        env.start_recording()
    env.reset()
    import time
    
    while True:
        # Step with zero action or some dummy action to let physics run
        # Action shape: (num_envs, num_agents, 3)
        # Using a small forward action to see movement
        action = torch.tensor([[[1.0, 0, 0]] * env.num_agents], dtype=torch.float32, device="cuda").repeat(env.num_envs, 1, 1)
        obs, _, done, _ = env.step(action)
        
        # Optional: Sleep to slow down if it's too fast, but usually sim handles it.
        # time.sleep(0.01)
