import abc
import numpy as np
from gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from collections import deque
import pandas as pd
from pylab import plt, mpl
from copy import deepcopy

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'



class ActionSpace:
    """
    Agent's exploration policy
    Simple uniform random policy between each action possible
    """

    def __init__(self, n):
        self.n = n

    def sample(self) -> int:
        return np.random.randint(0, self.n)


class Agent(metaclass=abc.ABCMeta):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
            learning_agent: bool = None,
            episodes: int = 1000,
            epsilon: float = 0.7,
            epsilon_min: float = 0.0001,
            epsilon_decay: float = 0.98,
            gamma: float = 0.97,
            batch_size: int = 1024,
    ):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.learning_agent = learning_agent
        self.episodes = episodes
        self._set_learning_args(epsilon, epsilon_min, epsilon_decay, gamma, batch_size)
        self.num_actions = 9
        self.step_info_per_episode = dict(map(lambda i: (i, None), range(1, self.episodes + 1)))
        self.step_info_per_eval_episode = self.step_info_per_episode.copy()
        self.done_info = {'nd_pnl':[], 'map':[], 'aum':[], 'depth':[]}
        self.done_info_eval = deepcopy(self.done_info)
        self.len_learn = None
        self.len_val = None

    def _set_learning_args(self, epsilon, epsilon_min, epsilon_decay, gamma, batch_size):
        if self.learning_agent:
            self.epsilon = epsilon  # Initial exploration rate
            self.epsilon_min = epsilon_min  # Minimum exploration rate
            self.epsilon_decay = epsilon_decay  # Decay rate for exploration rate, interval must be a pos function of the nb of xp
            self.gamma = gamma  # Discount factor for delayed reward
            self.batch_size = batch_size  # Batch size for replay
            self.memory = deque(maxlen=int(10e7)) #deque collection for limited history to train agent

        else:
            self.episodes = 1

    @property
    def actions(self) -> list:
        return list(range(self.num_actions))

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(len(self.actions))

    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        pass

    def _greedy_policy(self, state: np.ndarray):
        if np.random.random() <= self.epsilon:
            return self.action_space.sample()
        return self.get_action(state)

    def _play_one_step(self, state: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Method to play with the action at each step
        Each step of is composed of 3 elements: a state, the resulting reward, and finally a Boolean indicating
        whether the episode ended at that point (done = True)
        """
        action = self._greedy_policy(state) if self.learning_agent else self.get_action(state)
        next_state, reward, done, info = self.learn_env.step(action) # get the resulting state and reward
        if self.learning_agent:
            self.memory.append(
                [state, action, reward, next_state, done])  # Storing the resulting experience in the replay buffer
        return next_state, done

    def _compute_done(self, info, episode: int, done_info: dict):
        info = info[episode]
        done_info['nd_pnl'].append(info.nd_pnl)
        done_info['map'].append(info.map)
        done_info['aum'].append(info.aum)
        bar = len(info.inventories)
        done_info['depth'].append(bar)
        templ = '\nepisode: {:2d}/{} | bar: {:2d}/{}\n'
        templ += 'normalised pnl: {:5.2f} | mean abs position: {:5.2f}\n'
        templ += 'asset under management: {:5.2f} | success: {} \n'
        if done_info is self.done_info:
            success = True if bar == round(self.len_learn) else False
            print(templ.format(episode, self.episodes, bar, round(self.len_learn), info.nd_pnl, info.map, info.aum, success))
        else:
            success = True if bar == round(self.len_val) else False
            print(templ.format(episode, self.episodes, bar, round(self.len_val), info.nd_pnl, info.map, info.aum, success))
        print(50 * '*')

    def _replay(self):
        pass

    def learn(self, save: bool = False):
        for episode in range(1, self.episodes + 1):
            print(50 * '*')
            print(f'           Training of {self.get_name()}      ')
            state = self.learn_env.reset()
            print(f'    Start of trading: {self.learn_env.state.now_is} ')
            self.len_learn = (self.learn_env.end_of_trading - self.learn_env.state.now_is) / self.learn_env.step_size
            while self.learn_env.end_of_trading >= self.learn_env.state.now_is:
                state, done = self._play_one_step(state)
                if done:
                    self.step_info_per_episode[episode] = self.learn_env.info_calculator
                    self._compute_done(self.step_info_per_episode, episode, self.done_info)
                    break
            self._validate(episode)
            if self.learning_agent and len(self.memory) > self.batch_size:
                self._replay()
            self.graph_per_episode('pnls', episode)
            self.graph_per_episode('inventories', episode)
            self.info_reward(episode)
            self.graph_final('map') #'nd_pnl', 'map', 'aum', 'depth'
        if save: self.set_args()

    def _validate(self, episode: int):
        """
        Method to validate the performance of the DQL agent.
        """
        print(50 * '*')
        print(f'          Validation of {self.get_name()}      ')
        state = self.valid_env.reset()
        print(f'    Start of trading: {self.learn_env.state.now_is} ')
        self.len_val = (self.valid_env.end_of_trading - self.valid_env.state.now_is) / self.valid_env.step_size
        while self.valid_env.end_of_trading >= self.valid_env.state.now_is:
            action = self._greedy_policy(state) if self.learning_agent else self.get_action(state)
            state, reward, done, info = self.valid_env.step(action)
            if done:
                self.step_info_per_eval_episode[episode] = self.valid_env.info_calculator
                self._compute_done(self.step_info_per_eval_episode, episode, self.done_info_eval)
                break

    def graph_per_episode(self, metric: str=None, episode: int=None):
        train_metric = self.step_info_per_episode[episode].__dict__[metric]
        val_metric = self.step_info_per_eval_episode[episode].__dict__[metric]
        metrics = pd.DataFrame([train_metric, val_metric], index=['training','validation']).T
        metrics.plot(ylabel=metric, xlabel='n-th bar reaches', title=f'{metric} through time for episode {episode}')

    def info_reward(self, episode):
        train_reward = np.diff(self.step_info_per_episode[episode].__dict__['pnls'])
        val_reward = np.diff(self.step_info_per_eval_episode[episode].__dict__['pnls'])
        rewards = pd.DataFrame([train_reward, val_reward], index=['training','validation']).T
        stats = pd.DataFrame(rewards).describe()
        stats = np.round(stats)
        stats = stats.astype(int)
        rewards.plot.hist(title='Rewards histogram')
        rewards.rolling(100).mean().plot()
        rewards.rolling(100).std().plot()

    def graph_final(self, metric: str ):
        metrics = pd.DataFrame([self.done_info[metric], self.done_info_eval[metric]], index=['training','validation']).T
        metrics.plot(ylabel=metric, xlabel='n-th episodes', title=f'{metric} through episodes')








