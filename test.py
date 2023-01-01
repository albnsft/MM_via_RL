from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
import argparse
from helpers.main_helper import (
    add_env_args,
    get_env_configs,
)
from gym.utils import (
    get_reward_function,
    env_creator
)
from agents.baseline_agents import FixedActionAgent
from database.HistoricalDatabase import HistoricalDatabase


from utils.utils import read_data, split_dates

from orderbook.helpers import convert_orderbook_to_dataframe, convert_to_lobster_format
from simulation.OrderbookSimulator import OrderbookSimulator

"""
The k-th row in the 'message' file describes the limit order event causing the change in the limit
order book from line k-1 to line k in the 'orderbook' file
"""

if __name__ == '__main__':
    ticker = "MSFT"
    step_in_sec = 0.1
    date = datetime(2012, 6, 21)

    all_messages, all_books = read_data(ticker)
    dates = split_dates(split=0.75, date=date, hour_start=10, hour_end=10.10, step_in_sec=step_in_sec)


    db = HistoricalDatabase()
    simulator = OrderbookSimulator()

    start_of_episode = db.get_next_snapshot(datetime(2012, 6, 21, 9, 30), ticker).name
    start_of_episode += timedelta(microseconds=10 ** 6 - start_of_episode.microsecond)
    end_of_episode = start_of_episode + timedelta(seconds=5)
    simulator.reset_episode(start_of_episode)
    expected_book = db.get_last_snapshot(start_of_episode, ticker).to_dict()
    actual_book = convert_to_lobster_format(simulator.exchange.central_orderbook)
    assert(expected_book==actual_book)

    time_1 = start_of_episode + timedelta(seconds=1)
    simulator.forward_step(time_1)
    expected_book = db.get_last_snapshot(time_1, ticker).to_dict()
    actual_book = convert_to_lobster_format(simulator.exchange.central_orderbook)
    keys_to_drop = expected_book.keys() - actual_book.keys()
    for key in keys_to_drop:
        expected_book.pop(key)
    assert (expected_book == actual_book)

    from typing import cast
    from orderbook.create_order import create_order
    from orderbook.models import OrderDict
    min_buy_price_0 = simulator.min_buy_price
    orderbook = simulator.exchange.central_orderbook
    worst_buy_price = min(orderbook.buy.keys())
    worst_sell_price = max(orderbook.sell.keys())
    worst_buy_order_dict = cast(OrderDict, orderbook.buy[worst_buy_price][0].__dict__)
    worst_sell_order_dict = cast(OrderDict, orderbook.sell[worst_sell_price][0].__dict__)
    worst_buy_order_dict["volume"] = 100
    worst_sell_order_dict["volume"] = 200
    worst_buy_order_dict["is_external"] = False
    worst_sell_order_dict["is_external"] = False
    worst_buy_order_dict["price"] -= 1
    internal_buy = create_order("limit", worst_buy_order_dict)
    internal_sell = create_order("limit", worst_sell_order_dict)
    simulator.forward_step(end_of_episode, internal_orders=[internal_buy, internal_sell])

    internal_level = simulator.exchange.internal_orderbook.buy[worst_buy_order_dict["price"]]
    external_level = simulator.exchange.central_orderbook.buy[worst_buy_order_dict["price"]]





















    parser = argparse.ArgumentParser(description="")
    add_env_args(parser, dates, step_in_sec)
    #add_ray_args(parser)

    args = vars(parser.parse_args())
    train_env_config, eval_env_config = get_env_configs(args)
    train_env = env_creator(train_env_config)
    eval_env = env_creator(eval_env_config)

    agent = FixedActionAgent(0, train_env, eval_env)
    agent.learn()


    print('end')

