import os
import json
import shutil
import pickle
import neat

from source.core import Agent


class GeneticAlgorithm(Agent):
    def __init__(self, config: dict):
        super().__init__(config)

    def create_dir(self):
        super().create_dir()
        ga_config_path = self.config["ga_config_path"]
        if not os.path.isfile(ga_config_path):
            raise FileNotFoundError(f"NEAT config file not found: {ga_config_path}")

        # copy neat_config using shutil
        shutil.copy(
            ga_config_path,
            os.path.join(self.save_dir, os.path.basename(ga_config_path)),
        )
        return

    def make_env(self):
        raise NotImplementedError("make_env method must be implemented in subclass")

    def eval_genomes(self, genomes, config):
        raise NotImplementedError("eval_genomes method must be implemented in subclass")

    def test_winner(self, genome, config, episodes: int, steps: int):
        raise NotImplementedError("test_winner method must be implemented in subclass")

    def get_reporters(self, config):
        raise NotImplementedError(
            "get_reporters method must be implemented in subclass"
        )

    def stats_visualize(self, state, save_dir: str):
        raise NotImplementedError(
            "state_visualize method must be implemented in subclass"
        )

    def learn(self):
        self.create_dir()

        ga_config_path = self.config["ga_config_path"]

        self.env_id = self.config["env_id"]
        self.env_num = self.config["env_num"]
        self.num_episodes = self.config["num_episodes"]
        self.max_step = self.config["max_step"]

        num_generations = self.config["num_generations"]

        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            ga_config_path,
        )

        p = neat.Population(neat_config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        for r in self.get_reporters(neat_config):
            p.add_reporter(r)
        self.logger.info(f"NEAT config:\n{neat_config}")

        winner_genome = p.run(self.eval_genomes, num_generations)
        self.logger.info("\nBest genome:\n{!s}".format(winner_genome))

        data = {}
        data["config"] = neat_config
        data["winner_genome"] = winner_genome

        with open(os.path.join(self.save_dir, "best_genome.pkl"), "wb") as f:
            pickle.dump(data, f)
        self.logger.info(f"Best genome saved to 'best_genome.pkl' in {self.save_dir}'")

        # state visualize
        self.stats_visualize(stats, self.save_dir)

        # self.test_winner(winner, neat_config, test_episodes, test_steps)

    def load_winner_net(
        self,
        path: str,
        config,
    ):
        with open(path, "rb") as f:
            data = pickle.load(f)
        genome = data["winner_genome"]
        config = data["config"]

        self.winner_net = neat.nn.FeedForwardNetwork.create(genome, config)
        # self.logger.info(f"Loaded winner genome from {path}")
        print(f"✅ Loaded winner genome from {path}")

    def predict(self, obs):
        if self.winner_net is None:
            raise ValueError("Winner genome is not loaded. Call load_winner() first.")

        output = self.winner_net.activate(obs)
        action = output.index(max(output))

        return action
