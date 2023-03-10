from datetime import datetime
import argparse
from helpers.main_helper import (
    add_env_args,
    get_env_configs,
)
from mygym.utils import env_creator
from agents.baseline_agents import FixedActionAgent, RandomAgent, LstmAgent, DnnAgent
from utils.utils import split_dates

if __name__ == '__main__':

    ticker = "MSFT"
    step_in_sec = 1
    lags = 100
    date = datetime(2012, 6, 21)

    dates = split_dates(split=0.75, date=date, hour_start=10, hour_end=15.5, step_in_sec=step_in_sec)

    parser = argparse.ArgumentParser(description="")
    add_env_args(parser, dates, step_in_sec, lags, max_inv=9999)

    args = vars(parser.parse_args())

    """
    for i in range(9):
        train_env_config, eval_env_config = get_env_configs(args)
        train_env = env_creator(train_env_config)
        eval_env = env_creator(eval_env_config)
        agent = FixedActionAgent(i, train_env, eval_env)
        agent.learn()
    """

    train_env_config, eval_env_config = get_env_configs(args)
    train_env = env_creator(train_env_config)
    eval_env = env_creator(eval_env_config)
    agent = LstmAgent(train_env, eval_env)
    agent.learn()


    print('end')