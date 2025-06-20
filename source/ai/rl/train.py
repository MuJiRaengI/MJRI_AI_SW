import os
import sys

sys.path.append(os.path.abspath("."))


class Trainer:
    def __init__(self, agent, env, episodes=1000):
        self.agent = agent
        self.env = env
        self.episodes = episodes

    def train(self):
        self.agent.learn(total_timesteps=self.episodes)
        self.agent.save("ppo_cartpole")

        # for episode in range(self.episodes):
        #     state = self.env.reset()
        #     done = False
        #     total_reward = 0

        #     while not done:
        #         action = self.agent.select_action(state)
        #         next_state, reward, terminated, truncated, info = self.env.step(action)
        #         done = terminated or truncated
        #         self.agent.learn(state, action, reward, next_state, done)
        #         state = next_state
        #         total_reward += reward

        #     print(
        #         f"Episode {episode + 1}/{self.episodes}, Total Reward: {total_reward}"
        #     )


if __name__ == "__main__":
    import gym
    import pygame
    from stable_baselines3 import DQN, PPO

    pygame.init()
    pygame.display.set_caption(
        "Breakout Controller (A: Left, D: Right, 아무키도X: 정지)"
    )

    # env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = gym.make("CartPole-v1", render_mode="human")

    # 예시로 사용할 에이전트와 환경
    agent = PPO("MlpPolicy", env, verbose=1, device="cpu")

    trainer = Trainer(agent, env, episodes=10000)
    trainer.train()
