
from mqe.envs.go1.go1 import Go1
from mqe.envs.go1.go1_config import Go1Cfg

from mqe.envs.configs.go1_tug_config import Go1TugCfg
from mqe.envs.configs.go1_pushbox_config import Go1PushboxCfg
from mqe.envs.configs.go1_pushbox_light_config import Go1PushboxLightCfg
from mqe.envs.configs.go1_pushbox_medium_config import Go1PushboxMediumCfg
from mqe.envs.configs.go1_seesaw_config import Go1SeesawCfg
from mqe.envs.configs.go1_gate_config import Go1GateCfg
from mqe.envs.configs.go1_door_config import Go1DoorCfg
from mqe.envs.configs.go1_sheep_config import SingleSheepCfg, NineSheepCfg
from mqe.envs.configs.go1_football_config import Go1FootballDefenderCfg
from mqe.envs.configs.go1_plane_config import Go1PlaneCfg
from mqe.envs.configs.go1_wrestling_config import Go1WrestlingCfg
from mqe.envs.configs.go1_rotation_config import Go1RotationCfg
from mqe.envs.configs.go1_bridge_config import Go1BridgeCfg
from mqe.envs.configs.go1_pushball_config import Go1PushBallCfg
from mqe.envs.configs.go1_gate_with_button_config import Go1GateWithButtonCfg

from mqe.envs.wrappers.go1_pushbox_wrapper import Go1PushboxWrapper
from mqe.envs.wrappers.go1_tug_wrapper import Go1TugWrapper
from mqe.envs.wrappers.go1_seesaw_wrapper import Go1SeesawWrapper
from mqe.envs.wrappers.go1_gate_wrapper import Go1GateWrapper
# from mqe.envs.wrappers.go1_door_wrapper import Go1DoorWrapper
from mqe.envs.wrappers.go1_sheep_wrapper import Go1SheepWrapper
from mqe.envs.wrappers.go1_football_wrapper import Go1FootballDefenderWrapper as Go1FootballWrapper
from mqe.envs.wrappers.go1_wrestling_wrapper import Go1WrestlingWrapper
from mqe.envs.wrappers.go1_rotation_wrapper import Go1RotationWrapper
from mqe.envs.wrappers.go1_bridge_wrapper import Go1BridgeWrapper
from mqe.envs.wrappers.go1_pushball_wrapper import Go1PushBallWrapper
from mqe.envs.wrappers.go1_gate_with_button_wrapper import Go1GateWithButtonWrapper

from mqe.utils.task_registry import task_registry

task_registry.register( "go1tug", Go1, Go1TugCfg, Go1TugWrapper)
task_registry.register( "go1pushbox", Go1, Go1PushboxCfg, Go1PushboxWrapper)
task_registry.register( "go1pushbox_light", Go1, Go1PushboxLightCfg, Go1PushboxWrapper)
task_registry.register( "go1pushbox_medium", Go1, Go1PushboxMediumCfg, Go1PushboxWrapper)
task_registry.register( "go1seesaw", Go1, Go1SeesawCfg, Go1SeesawWrapper)
task_registry.register( "go1gate", Go1, Go1GateCfg, Go1GateWrapper)
# task_registry.register( "go1door", Go1, Go1DoorCfg, Go1DoorWrapper)
task_registry.register( "go1sheep", Go1, SingleSheepCfg, Go1SheepWrapper)
task_registry.register( "go1sheep-9", Go1, NineSheepCfg, Go1SheepWrapper)
task_registry.register( "go1football", Go1, Go1FootballDefenderCfg, Go1FootballWrapper)
task_registry.register( "go1plane", Go1, Go1PlaneCfg, Go1GateWrapper)
task_registry.register( "go1wrestling", Go1, Go1WrestlingCfg, Go1WrestlingWrapper)
task_registry.register( "go1rotation", Go1, Go1RotationCfg, Go1RotationWrapper)
task_registry.register( "go1bridge", Go1, Go1BridgeCfg, Go1BridgeWrapper)
task_registry.register( "go1pushball", Go1, Go1PushBallCfg, Go1PushBallWrapper)
task_registry.register( "go1gatewithbutton", Go1, Go1GateWithButtonCfg, Go1GateWithButtonWrapper)
