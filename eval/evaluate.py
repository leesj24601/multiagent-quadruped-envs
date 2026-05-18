import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PUSHBALL_SUCCESS_THRESHOLD = 0.2
GATEWITHBUTTON_SUCCESS_X = 4.0

RESULT_FIELDS = [
    "timestamp",
    "task",
    "algo",
    "checkpoint",
    "episodes",
    "max_steps",
    "success_count",
    "success_rate",
    "num_envs",
    "seed",
    "success_definition",
]


def is_pushball_success(
    ball_position: Sequence[float],
    target_position: Sequence[float],
    threshold: float = PUSHBALL_SUCCESS_THRESHOLD,
) -> bool:
    return math.dist([float(v) for v in ball_position], [float(v) for v in target_position]) < threshold


def is_gatewithbutton_success(
    agent_positions: Iterable[Sequence[float]],
    threshold_x: float = GATEWITHBUTTON_SUCCESS_X,
) -> bool:
    return any(float(position[0]) > threshold_x for position in agent_positions)


def parse_args():
    from openrl.configs.config import create_config_parser
    from openrl_ws.utils import parse_arguments

    parser = create_config_parser()
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "go1pushball",
            "choices": ["go1pushball", "go1gatewithbutton"],
            "help": "Task to evaluate.",
        },
        {
            "name": "--algo",
            "type": str,
            "default": "ppo",
            "choices": ["ppo", "mat"],
            "help": "Algorithm used by the checkpoint.",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": True,
            "help": "Path to the trained checkpoint.",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 1,
            "help": "Number of environments. Evaluation currently expects 1 for clean episode accounting.",
        },
        {
            "name": "--episodes",
            "type": int,
            "default": 100,
            "help": "Number of evaluation episodes.",
        },
        {
            "name": "--max_steps",
            "type": int,
            "default": None,
            "help": "Maximum steps per episode. Defaults to the task episode length.",
        },
        {
            "name": "--output",
            "type": str,
            "default": str(Path(__file__).resolve().parent / "results.csv"),
            "help": "CSV file where evaluation results are appended.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "help": "Run without viewer.",
        },
        {
            "name": "--render",
            "action": "store_true",
            "help": "Open the Isaac Gym viewer instead of running headless.",
        },
        {
            "name": "--record_video",
            "action": "store_true",
            "help": "Record video through the environment wrapper.",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "RL device.",
        },
        {
            "name": "--use_wandb",
            "action": "store_true",
            "help": "Unused in evaluation, kept for compatibility.",
        },
        {
            "name": "--use_tensorboard",
            "action": "store_true",
            "help": "Unused in evaluation, kept for compatibility.",
        },
        {
            "name": "--exp_name",
            "type": str,
            "default": "eval",
            "help": "Experiment name kept for compatibility.",
        },
    ]
    args = parse_arguments(parser, custom_parameters=custom_parameters)

    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"

    if args.algo == "ppo":
        args.config = "./openrl_ws/cfgs/ppo.yaml"
    elif args.algo == "mat":
        args.config = "./openrl_ws/cfgs/mat.yaml"

    args.headless = not getattr(args, "render", False)

    if args.task not in {"go1pushball", "go1gatewithbutton"}:
        raise ValueError("--task must be one of: go1pushball, go1gatewithbutton.")
    if args.algo not in {"ppo", "mat"}:
        raise ValueError("--algo must be one of: ppo, mat.")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required.")
    if args.num_envs != 1:
        raise ValueError("This evaluator currently requires --num_envs 1 for clean per-episode resets.")
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than 0.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max_steps must be greater than 0.")

    return args


def build_agent_and_env(args):
    import isaacgym  # noqa: F401
    from mqe.envs.utils import custom_cfg
    from openrl.modules.common import MATNet, PPONet
    from openrl.runners.common import MATAgent, PPOAgent
    from openrl_ws.utils import MATWrapper, make_env

    env, _ = make_env(args, custom_cfg(args))

    if args.algo == "ppo":
        net = PPONet(env, cfg=args, device=args.rl_device)
        agent = PPOAgent(net)
    elif args.algo == "mat":
        env = MATWrapper(env)
        net = MATNet(env, cfg=args, device=args.rl_device)
        agent = MATAgent(net, use_wandb=args.use_wandb)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    agent.load(args.checkpoint)
    agent.set_env(env)
    return agent, env


def unwrap_task_env(env):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        class_name = current.__class__.__name__
        if class_name in {"Go1PushBallWrapper", "Go1GateWithButtonWrapper"}:
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("Could not find the task wrapper inside the environment.")


def success_definition(task: str) -> str:
    if task == "go1pushball":
        return f"ball_to_hole_3d_distance_lt_{PUSHBALL_SUCCESS_THRESHOLD}"
    if task == "go1gatewithbutton":
        return f"any_agent_x_gt_{GATEWITHBUTTON_SUCCESS_X}"
    raise ValueError(f"Unsupported task: {task}")


def compute_success(task: str, task_env) -> bool:
    import torch

    if task == "go1pushball":
        ball_position = task_env.root_states_npc[:, :3] - task_env.env.env_origins
        target_position = task_env.hole_pos[:, 0, :]
        return bool((torch.norm(ball_position - target_position, dim=1)[0] < PUSHBALL_SUCCESS_THRESHOLD).item())

    if task == "go1gatewithbutton":
        agent_positions = task_env.env.base_pos.view(
            task_env.env.num_envs,
            task_env.env.num_agents,
            3,
        )
        agent_positions = agent_positions - task_env.env.env_origins.unsqueeze(1)
        return bool(torch.any(agent_positions[0, :, 0] > GATEWITHBUTTON_SUCCESS_X).item())

    raise ValueError(f"Unsupported task: {task}")


def env_done(done) -> bool:
    import numpy as np

    done_array = np.asarray(done)
    return bool(done_array.reshape(done_array.shape[0], -1).all(axis=1)[0])


def get_max_steps(args, task_env) -> int:
    if args.max_steps is not None:
        return int(args.max_steps)
    return int(getattr(task_env.env, "max_episode_length", 1000))


def evaluate(args):
    agent, env = build_agent_and_env(args)
    task_env = unwrap_task_env(env)
    max_steps = get_max_steps(args, task_env)

    success_count = 0
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            episode_success = False

            for _ in range(max_steps):
                action, _ = agent.act(obs)
                obs, _, done, _ = env.step(action)

                episode_success = compute_success(args.task, task_env)
                if episode_success or env_done(done):
                    break

            if episode_success:
                success_count += 1

            print(
                f"[{episode + 1}/{args.episodes}] "
                f"success={int(episode_success)} "
                f"running_success_rate={success_count / (episode + 1):.3f}"
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except AttributeError as exc:
                print(f"Warning: ignored environment close error: {exc}", file=sys.stderr)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": args.task,
        "algo": args.algo,
        "checkpoint": str(Path(args.checkpoint)),
        "episodes": args.episodes,
        "max_steps": max_steps,
        "success_count": success_count,
        "success_rate": f"{success_count / args.episodes:.6f}",
        "num_envs": args.num_envs,
        "seed": getattr(args, "seed", ""),
        "success_definition": success_definition(args.task),
    }


def append_result(output_path: Path, result: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(result)


def main():
    args = parse_args()
    result = evaluate(args)
    output_path = Path(args.output)
    append_result(output_path, result)

    print(
        "Evaluation complete: "
        f"{result['success_count']}/{result['episodes']} "
        f"success_rate={result['success_rate']} "
        f"saved_to={output_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
