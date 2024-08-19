import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

# Create the environment
env = gym.make('CartPole-v1')

# Vectorize the environment for parallel training
env = make_vec_env(lambda: gym.make('CartPole-v1'), n_envs=4)

# Define the model using PPO
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

    # If any of the environments are done, reset them
    if dones.any():  # Use .any() for vectorized environments
        obs = env.reset()

# Save the model
model.save("ppo_cartpole")

# Load the model
model = PPO.load("ppo_cartpole")

# Example: Plotting the reward over episodes
rewards = []
for i in range(100):
    obs = env.reset()
    episode_reward = 0
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]  # Since reward is an array, take the first element
        if done.any():  # Use .any() for vectorized environments
            break
    rewards.append(episode_reward)

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

