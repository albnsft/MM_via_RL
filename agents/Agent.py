import abc
import numpy as np
from mygym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from mygym.utils import plot_per_episode, plot_final
from collections import deque
from pylab import plt, mpl
from copy import deepcopy
import time

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
            episodes: int = None,
            epsilon: float = None,
            epsilon_min: float = None,
            epsilon_decay: float = None,
            gamma: float = None,
            batch_size: int = None,
            seed: int = 42
    ):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.learning_agent = learning_agent
        self.episodes = episodes
        self.seed = seed
        self._set_learning_args(epsilon, epsilon_min, epsilon_decay, gamma, batch_size)
        self.num_actions = 9
        self.step_info_per_episode = dict(map(lambda i: (i, None), range(1, self.episodes + 1)))
        self.step_info_per_eval_episode = deepcopy(self.step_info_per_episode)
        self.done_info = {'nd_pnl': [], 'map': [], 'aum': [], 'depth': []}
        self.done_info_eval = deepcopy(self.done_info)
        self.len_learn = None
        self.len_val = None

    def _set_seed_np(self):
        np.random.seed(self.seed)

    def _set_learning_args(self, epsilon: float, epsilon_min: float, epsilon_decay: float, gamma: float,
                           batch_size: int):
        self._set_seed_np()
        if self.learning_agent:
            self.epsilon = epsilon  # Initial exploration rate
            self.epsilon_min = epsilon_min  # Minimum exploration rate
            self.epsilon_decay = epsilon_decay  # Decay rate for exploration rate, interval must be a pos function of the nb of xp
            self.gamma = gamma  # Discount factor for delayed reward
            self.batch_size = batch_size  # Batch size for replay
            self.memory = deque(maxlen=int(10e4))  # deque collection for limited history to train agent
        else:
            self.episodes = 1
            self.learn_env.threshold = 1
            self.valid_env.threshold = 1
        if self.learn_env.max_inventory >= 10000:
            self.learn_env.market_order_fraction_of_inventory = 0  # no market order clearing
            self.learn_env.market_order_clearing = False

    @property
    def actions(self):
        return list(range(self.num_actions))

    @property
    def action_space(self):
        return ActionSpace(len(self.actions))

    @abc.abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    def _greedy_policy(self, state: np.ndarray):
        if np.random.random() <= self.epsilon:
            return self.action_space.sample()
        return self.get_action(state)

    def _play_one_step(self, state: np.ndarray):
        """
        Method to play with the action at each step
        Each step of is composed of 3 elements: a state, the resulting reward, and finally a Boolean indicating
        whether the episode ended at that point (done = True)
        """
        action = self._greedy_policy(state) if self.learning_agent else self.get_action(state)
        next_state, reward, done, info = self.learn_env.step(action)  # get the resulting state and reward
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
        if (episode-1) % 10 == 0:
            templ = '\nepisode: {:2d}/{} | bar: {:2d}/{} | epsilon: {:5.2f}\n'
            templ += 'normalised pnl: {:5.2f} | mean abs position: {:5.2f}\n'
            templ += 'asset under management: {:5.2f} | success: {} \n'
            if done_info is self.done_info:
                print(50 * '*')
                print(f'           Training of {self.get_name()}      ')
                print(f'    Start of trading: {self.learn_env.start_of_trading} ')
                if bar > round(self.len_learn): bar = round(self.len_learn)
                success = True if bar == round(self.len_learn) else False
                print(templ.format(episode, self.episodes, bar, round(self.len_learn), self.epsilon,
                                   info.nd_pnl, info.map, info.aum, success))
            else:
                print(f'          Validation of {self.get_name()}      ')
                print(f'    Start of trading: {self.valid_env.start_of_trading} ')
                if bar > round(self.len_val): bar = round(self.len_val)
                success = True if bar == round(self.len_val) else False
                print(templ.format(episode, self.episodes, bar, round(self.len_val), self.epsilon,
                                   info.nd_pnl, info.map, info.aum, success))
                print(50 * '*')

    @abc.abstractmethod
    def replay(self):
        pass

    def learn(self, save: bool = False):
        start = time.time()
        for episode in range(1, self.episodes + 1):
            state = self.learn_env.reset()
            self.len_learn = (self.learn_env.end_of_trading - self.learn_env.state.now_is) / self.learn_env.step_size
            while self.learn_env.end_of_trading >= self.learn_env.state.now_is:
                state, done = self._play_one_step(state)
                if done:
                    self.step_info_per_episode[episode] = self.learn_env.info_calculator
                    self._compute_done(self.step_info_per_episode, episode, self.done_info)
                    break
            self._validate(episode)
            if self.learning_agent and len(self.memory) > self.batch_size:
                self.replay()
            if episode % 10 == 0:
                plot_per_episode(self.learn_env.ticker, self.get_name(),
                                 self.learn_env.step_size, self.learn_env.market_order_fraction_of_inventory,
                                 self.learn_env.per_step_reward_function_midprice, self.step_info_per_episode,
                                 self.step_info_per_eval_episode, episode, self.done_info, self.done_info_eval)
        if self.episodes > 1:
            plot_final(self.done_info, self.done_info_eval, self.learn_env.ticker, self.get_name(),
                       self.learn_env.step_size, self.learn_env.market_order_fraction_of_inventory,
                       self.learn_env.per_step_reward_function_midprice)
        print(f'Time elapsed: {round((time.time() - start) / 3600, 3)} hours')
        if save: self.set_args()

    def _validate(self, episode: int):
        """
        Method to validate the performance of the DQL agent.
        only relies on the exploitation of the currently optimal policy
        """
        state = self.valid_env.reset()
        self.len_val = (self.valid_env.end_of_trading - self.valid_env.state.now_is) / self.valid_env.step_size
        while self.valid_env.end_of_trading >= self.valid_env.state.now_is:
            action = self.get_action(state)
            state, reward, done, info = self.valid_env.step(action)
            if done:
                self.step_info_per_eval_episode[episode] = self.valid_env.info_calculator
                self._compute_done(self.step_info_per_eval_episode, episode, self.done_info_eval)
                break
