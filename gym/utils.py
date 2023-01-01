from datetime import timedelta

from database.HistoricalDatabase import HistoricalDatabase
from simulation.OrderbookSimulator import OrderbookSimulator
from gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from rewards.RewardFunctions import InventoryAdjustedPnL, PnL

from features.Features import Portfolio


def get_reward_function(reward_function: str, inventory_aversion: float = 0.1):
    if reward_function == "AD":  # asymmetrically dampened
        return InventoryAdjustedPnL(inventory_aversion=inventory_aversion, asymmetrically_dampened=True)
    elif reward_function == "SD":  # symmetrically dampened
        return InventoryAdjustedPnL(inventory_aversion=inventory_aversion, asymmetrically_dampened=False)
    elif reward_function == "PnL":
        return PnL()
    else:
        raise NotImplementedError("You must specify one of 'AS', 'SD', 'PnL'")


def env_creator(env_config, database: HistoricalDatabase = HistoricalDatabase()):

    if env_config["features"] == "agent_state":
        features = HistoricalOrderbookEnvironment.get_default_features(
            step_size=timedelta(seconds=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )[-3:]

    elif env_config["features"] == "market_state":
        features = HistoricalOrderbookEnvironment.get_default_features(
            step_size=timedelta(seconds=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )[:-3]

    elif env_config["features"] == "full_state":
        features = HistoricalOrderbookEnvironment.get_default_features(
            step_size=timedelta(seconds=env_config["step_size"]),
            normalisation_on=env_config["normalisation_on"],
        )

    orderbook_simulator = OrderbookSimulator(
        ticker=env_config["ticker"],
        database=database,
    )
    env = HistoricalOrderbookEnvironment(
        start_of_trading=env_config["start_trading"],
        end_of_trading=env_config["end_trading"],
        max_inventory=env_config["max_inventory"],
        ticker=env_config["ticker"],
        simulator=orderbook_simulator,
        features=features,
        step_size=timedelta(seconds=env_config["step_size"]),
        market_order_fraction_of_inventory=env_config["market_order_fraction_of_inventory"],
        initial_portfolio=Portfolio(inventory=env_config["initial_inventory"], cash=env_config["initial_cash"],
                                    gain=env_config["initial_gain"]),
        per_step_reward_function=get_reward_function(env_config["per_step_reward_function"],
                                                     env_config["inventory_aversion"]),
        n_lags_feature=env_config["n_lags_feature"]
    )
    return env