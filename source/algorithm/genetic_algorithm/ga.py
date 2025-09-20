import os
import json
import shutil
import pickle
import neat

from source.algorithm.algorithm import Algorithm


class GeneticAlgorithm(Algorithm):
    def __init__(
        self,
        save_dir: str,
        logging_freq=100,
        detailed_logging_freq=500,
    ):
        super().__init__(save_dir, logging_freq, detailed_logging_freq)
        self.env_id = None
        self.env_num = None
        self.num_episodes = None
        self.max_step = None
        self.winner_net = None

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

    def learn(self, config: dict):
        save_config_path = os.path.join(self.save_dir, "config.json")
        with open(save_config_path, "w") as f:
            json.dump(config, f, indent=4)

        self.env_id = config["env_name"]
        self.env_num = config["env_num"]
        self.num_episodes = config["num_episodes"]
        self.max_step = config["max_step"]

        neat_config_path = config["neat_config_path"]
        num_generations = config["num_generations"]

        # copy neat_config using shutil
        shutil.copy(
            neat_config_path,
            os.path.join(self.save_dir, os.path.basename(neat_config_path)),
        )

        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_path,
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
        self.logger.info(f"Loaded winner genome from {path}")

    def predict(self, obs):
        if self.winner_net is None:
            raise ValueError("Winner genome is not loaded. Call load_winner() first.")

        output = self.winner_net.activate(obs)
        action = output.index(max(output))

        return action
