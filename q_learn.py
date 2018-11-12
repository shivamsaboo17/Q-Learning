import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm


class QLearn:

    def __init__(self, environment, total_episodes=1000, learning_rate=0.1, max_steps=100, discount_factor=0.9, decay_rate=0.005, epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01):
        self.env = gym.make(environment)
        self.q_table = self._build_q_table()
        self.total_episodes = total_episodes
        self.learning_rate = 0.1
        self.max_steps = 100
        self.discount_factor = discount_factor
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.rewards = []
        self.epsilons = []

    def _iterate(self, state, done, mode='train'):
        total_rewards = 0
        for _ in range(self.max_steps):
            if mode == 'train':
                explore = random.uniform(0, 1) < self.epsilon
                if explore:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state, :])
            else:
                action = np.argmax(self.q_table[state, :])
            new_state, reward, done, _ = self.env.step(action)
            if mode == 'train':
                delta_reward = reward + self.discount_factor * np.max(self.q_table[new_state, :]) - self.q_table[state, action]
                self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * delta_reward
            state = new_state
            total_rewards += reward
            if done:
                if mode == 'test':
                    self.env.render()
                break
        return total_rewards

    def save(self, filename):
        state_dict = self.prepare_state_dict()
        with open(filename, 'wb') as f:
            pkl.dump(state_dict, f)

    def load_state_dict(self, filename):
        with open(filename, 'rb') as f:
            state_dict = pkl.load(f)
            self.check_loaded_model(state_dict)

    def prepare_state_dict(self):
        return {'Model':self.q_table, 'Epsilon':self.epsilon, 'Max Epsilon':self.epsilon}

    def check_loaded_model(self, state_dict):
        self.q_table = state_dict['Model']
        if self.q_table.shape != (self.env.observation_space.n, self.env.action_space.n):
            raise ValueError('Model-Environment mismatch')
        self.epsilon = state_dict['Epsilon']
        self.max_epsilon = state_dict['Max Epsilon']
        print('Loaded model successfully and overriden epsilon.')
        pass

    def plot(self, epsilon=False):
        if epsilon:
            plt.plot(self.epsilons)
        else:
            plt.plot(self.rewards)
        plt.show()

    def fit(self):
        self.rewards = []
        with tqdm(total=self.total_episodes) as pbar:
            for i in range(self.total_episodes):
                state = self.env.reset()
                total_reward = self._iterate(state, False, mode='train')
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*i)
                self.epsilons.append(self.epsilon) 
                self.rewards.append(total_reward)
                pbar.set_description('Total Reward: %f' %(total_reward))
                pbar.update(1)

    def test(self, test_episodes):
        self.rewards = []
        for _ in range(test_episodes):
            total_reward = self._iterate(self.env.reset(), False, mode='test')
            self.rewards.append(total_reward)
        print('Average Score is', np.mean(np.array(self.rewards)))
        

    def _build_q_table(self):
        actions_sz = self.env.action_space.n
        states_sz = self.env.observation_space.n
        return np.zeros(shape=(states_sz, actions_sz))
