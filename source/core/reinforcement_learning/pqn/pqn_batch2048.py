from source.envs import Batch2048EnvFast
from source.core.reinforcement_learning.pqn import PQN


class PQNBatch2048(PQN):
    def __init__(self, config: dict):
        super().__init__(config)
