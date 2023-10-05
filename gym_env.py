import gym
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500

# print(env.action_space.n)
# print(env.observation_space.high)
# print(env.observation_space.low)

DISCRETE_OS_SIZE = [20, 20] # [20] * len(env.observation_space.high)
#discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
high = np.array([0.6, 0.07], dtype=np.float32)
low = np.array([-1.2, -0.07], dtype=np.float32)
discrete_os_win_size = (high - low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [3,]))

ep_rewards = []
aggr_ep_rewards = {"ep": [], "avg": [], "min": [], "max": []}

def get_discrete_state(state):
    discrete_state = (state - low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print("Episode:", episode)
        env = gym.make("MountainCar-v0", render_mode="human")
    else:
        pass
        env = gym.make("MountainCar-v0", render_mode="None")

    episode_reward = 0

    terminated, truncated = [False, False]
    discrete_state = get_discrete_state(env.reset()[0])

    while not terminated and not truncated:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if not terminated and truncated:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"Challenge is complete on episode {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["min"].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards["max"].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode}, avg: {average_reward}, min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}")
    env.close()

plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="avg")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="min")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="max")
plt.legend(loc=4)
plt.show()