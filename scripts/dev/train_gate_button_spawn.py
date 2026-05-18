import isaacgym  # noqa: F401

from openrl.utils.logger import Logger
from openrl_ws.utils import MATWrapper, get_args, make_env
from mqe.envs.utils import custom_cfg


BUTTON_AGENT_ID = 1
SPAWN_Z = 0.42


def _agent_origin_relative_to_env(cfg, agent_id):
    terrain = cfg.terrain.BarrierTrack_kwargs
    init = terrain["init"]
    track_width = terrain["track_width"]
    room_x, room_y = init["room_size"]
    border = init["border_width"]
    offset_x, offset_y = init["offset"]
    num_agents = cfg.env.num_agents

    room_y_total = room_y * num_agents + border * (num_agents - 1)
    room_origin_y = (track_width - room_y_total) / 2.0 + offset_y

    origin_x = init["block_length"] / 2.0 + offset_x
    origin_y = -track_width / 2.0
    origin_y += room_origin_y + agent_id * (room_y + border) + room_y / 2.0

    return origin_x, origin_y


def gate_button_spawn_cfg(args):
    base_custom_cfg = custom_cfg(args)

    def fn(cfg):
        cfg = base_custom_cfg(cfg)
        cfg.domain_rand.init_base_pos_range = dict(x=[0.0, 0.0], y=[0.0, 0.0])

        button_x, button_y = cfg.game.button_pos
        origin_x, origin_y = _agent_origin_relative_to_env(cfg, BUTTON_AGENT_ID)
        cfg.init_state.init_states[BUTTON_AGENT_ID].pos = [
            button_x - origin_x,
            button_y - origin_y,
            SPAWN_Z,
        ]

        print(
            "[button-spawn-config] "
            f"agent_id={BUTTON_AGENT_ID} "
            f"button={cfg.game.button_pos} "
            f"agent_origin_rel={[origin_x, origin_y]} "
            f"init_pos={cfg.init_state.init_states[BUTTON_AGENT_ID].pos}"
        )
        return cfg

    return fn


def train(args):
    args.task = "go1gatewithbutton"

    if args.algo == "sppo" or args.algo == "dppo":
        single_agent = True
    else:
        single_agent = False

    env, _ = make_env(args, gate_button_spawn_cfg(args), single_agent)

    if args.algo == "ppo":
        args.config = "./openrl_ws/cfgs/ppo.yaml"

    elif args.algo == "jrpo":
        args.config = "./openrl_ws/cfgs/jrpo.yaml"

    elif args.algo == "mat":
        args.config = "./openrl_ws/cfgs/mat.yaml"

        from openrl.modules.common import MATNet
        from openrl.runners.common import MATAgent

        env = MATWrapper(env)
        net = MATNet(env, cfg=args, device=args.rl_device)
        agent = MATAgent(net, use_wandb=args.use_wandb)

    elif args.algo == "sppo" or args.algo == "dppo":
        pass

    else:
        raise NotImplementedError

    if "po" in args.algo:
        from openrl.modules.common import PPONet
        from openrl.runners.common import PPOAgent

        net = PPONet(env, cfg=args, device=args.rl_device)
        agent = PPOAgent(net)
        logger = Logger(
            cfg=net.cfg,
            project_name="MQE",
            scenario_name=args.task,
            wandb_entity="ziyanx02",
            exp_name=args.exp_name,
            log_path="./log",
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
        )
        if getattr(args, "checkpoint", None) is not None:
            agent.load(args.checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        agent.train(total_time_steps=args.train_timesteps, logger=logger)
    else:
        agent.train(total_time_steps=args.train_timesteps)

    if getattr(args, "exp_name", None):
        dir_name = "./checkpoints/" + args.task + "/" + args.exp_name
    else:
        dir_name = "./checkpoints/" + args.task
    agent.save(dir_name)


if __name__ == "__main__":
    train(get_args())
