from gymnasium.envs.registration import register
from simple_coverage.env import SimpleGridworld
from simple_coverage.complete_env import CompleteSimpleGridworld

register(
    id="standard",
    entry_point="simple_coverage:SimpleGridworld"
)

register(
    id="complete",
    entry_point="simple_coverage:CompleteSimpleGridworld"
)