import abc

import numpy as np
import random

from agents.Agent import Agent
from mygym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment

from agents.value_approximators.baseline_nets import Net
from agents.value_approximators.Nets import Params, LSTM, DNN

from torchsummary import summary


class RandomAgent(Agent):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
    ):
        super().__init__(learn_env, valid_env, False)

    def get_action(self, state: np.ndarray) -> int:
        return self.action_space.sample()

    def get_name(self):
        return "RandomAgent"


class FixedActionAgent(Agent):
    def __init__(
            self,
            fixed_action: int = None,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None
    ):
        super().__init__(learn_env, valid_env, False)
        self.fixed_action = fixed_action

    def get_action(self, state: np.ndarray) -> int:
        return self.fixed_action

    def get_name(self) -> str:
        return f"FixedAction_{self.fixed_action}"


class BasicAgent(Agent):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
            lmbda: float = None):
        super().__init__(learn_env, valid_env)
        self.lmbda = lmbda

    """
    This agent used a state representation comprising only of the agent-state and the non-dampened PnL reward function.
    This agent was trained using one step Q-learning and SARSA
    The addition of eligibility traces (known as Q(λ) and SARSA(λ)) improved the agents’ performance
    The basic agents represents the best attainable results using “standard techniques”
    """

    def get_action(self, state: np.ndarray) -> int:
        pass

    def get_name(self):
        return "BasicAgent"


class ConsolidateAgent(Agent):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
            lmbda: float = None):
        super().__init__(learn_env, valid_env)
        self.lmbda = lmbda

    """
    Consolidation of the best variants on the basic agent is considered. 
    It uses the asymmetrically dampened reward function with a LCTC state-space, trained using SARSA
    """

    def get_action(self, state: np.ndarray) -> int:
        pass

    def get_name(self):
        return "ConsolidatedAgent"


class BaseDQN(Agent):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
            episodes: int = 1000,
            epsilon: float = 0.95,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.99,
            gamma: float = 0.97,
            batch_size: int = 1024,
            type_algo: str = 'mask'
    ):
        super().__init__(learn_env, valid_env, True, episodes, epsilon, epsilon_min, epsilon_decay, gamma, batch_size)
        self.type_algo = type_algo
        self._set_seed_rand()

    def _set_seed_rand(self):
        random.seed(self.seed)

    @abc.abstractmethod
    def _compute_fit(self, state: np.ndarray, target, mask: np.ndarray = None):
        pass

    @abc.abstractmethod
    def _compute_prediction(self, state: np.ndarray, idmax: bool):
        pass

    def get_action(self, state: np.ndarray):
        """
        optimal policy is defined trivially as: when the agent is in state s, it will select the action with the highest
        value for that state
        """
        return self._compute_prediction(state, idmax=True)

    def replay(self):
        """
        Method to retrain the DQN model based on batches of memorized experiences
        Updating the policy function Q regularly, improve the learning considerably
        """
        batch = list(map(np.array, zip(*random.sample(self.memory, self.batch_size))))
        states, actions, rewards, next_states, dones = batch
        """
        approximate Q-Value(target) should be close to the reward the agent gets after playing action a in state s
        plus the future discounted value of playing optimally from then on
        """
        rewards += (1 - dones) * self.gamma * np.amax(self._compute_prediction(next_states, idmax=False).cpu().numpy(), axis=1)

        if self.type_algo != 'mask':
            all_Q_values = self._compute_prediction(states, idmax=False).cpu().numpy()
            all_Q_values_target = all_Q_values.copy()
            all_Q_values_target[range(len(actions)), actions] = rewards
            self._compute_fit(states, all_Q_values_target)
        else:
            masks = np.zeros((len(actions), len(self.actions)))
            masks[range(len(actions)), actions] = 1
            self._compute_fit(states, rewards, masks)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DnnAgent(BaseDQN):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
            hidden_dim: int = 256,
            n_hidden: int = 1,
            lr: float = 0.001,
            dropout: float = 0.2

    ):
        super().__init__(learn_env, valid_env)
        self._set_model(hidden_dim, n_hidden, lr, dropout)

    def get_name(self):
        return "DnnAgent"

    def _set_model(self, hidden_dim: int, n_hidden: int, lr: float, dropout: float):
        if n_hidden == 0: dropout = 0
        assert (self.learn_env.n_lags_feature == 0)
        params = Params(input_dim=len(self.learn_env.features), hidden_dim=hidden_dim, n_hidden=n_hidden,
                        dropout=dropout, seed=self.seed)
        self.model = Net(DNN(params), lr=lr, name=self.get_name(), seed=self.seed)
        summary(self.model.model, (1, len(self.learn_env.features)))

    def _compute_fit(self, state: np.ndarray, target, mask: np.ndarray = None):
        self.model.fit(state, target, mask)

    def _compute_prediction(self, state: np.ndarray, idmax: bool):
        return self.model.predict(state, idmax=idmax)


class LstmAgent(BaseDQN):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None,
            hidden_dim: int = 256,
            n_hidden: int = 1,
            lr: float = 0.001,
            dropout: float = 0.2

    ):
        super().__init__(learn_env, valid_env)
        self._set_model(hidden_dim, n_hidden, lr, dropout)

    def get_name(self):
        return "LstmAgent"

    def _set_model(self, hidden_dim: int, n_hidden: int, lr: float, dropout: float):
        if n_hidden == 0: dropout = 0
        params = Params(input_dim=len(self.learn_env.features), hidden_dim=hidden_dim, n_hidden=n_hidden,
                        dropout=dropout, seed=self.seed)
        self.model = Net(LSTM(params), lr=lr, name=self.get_name(), seed=self.seed)
        summary(self.model.model, (self.learn_env.n_lags_feature, len(self.learn_env.features)))

    def _compute_fit(self, state: np.ndarray, target, mask: np.ndarray = None):
        self.model.fit(state, target, mask)

    def _compute_prediction(self, state: np.ndarray, idmax: bool):
        return self.model.predict(state, idmax=idmax)
