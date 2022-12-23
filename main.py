from datetime import datetime
import argparse
from helpers.main_helper import (
    add_env_args,
    get_env_configs,
)
from gym.utils import env_creator
from agents.baseline_agents import FixedActionAgent
from utils.utils import split_dates

if __name__ == '__main__':

    ticker = "MSFT"
    step_in_sec = 0.1
    date = datetime(2012, 6, 21)

    dates = split_dates(split=0.75, date=date, hour_start=10, hour_end=10.10, step_in_sec=step_in_sec)

    parser = argparse.ArgumentParser(description="")
    add_env_args(parser, dates, step_in_sec)

    args = vars(parser.parse_args())
    train_env_config, eval_env_config = get_env_configs(args)
    train_env = env_creator(train_env_config)
    eval_env = env_creator(eval_env_config)

    agent = FixedActionAgent(0, train_env, eval_env)
    agent.learn()


    print('end')

