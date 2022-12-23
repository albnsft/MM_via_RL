import abc

import numpy as np

from agents.Agent import Agent
from gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment


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


class BaseDQN(Agent, metaclass=abc.ABCMeta):
    def __init__(
            self,
            learn_env: HistoricalOrderbookEnvironment = None,
            valid_env: HistoricalOrderbookEnvironment = None, ):
        super().__init__(learn_env, valid_env)

    @abc.abstractmethod
    def _set_model(self, hypermodel) -> None:
        pass

    @abc.abstractmethod
    def _compute_fit(self, state: np.ndarray, target: np.ndarray) -> None:
        pass

