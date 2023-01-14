from datetime import timedelta

from database.HistoricalDatabase import HistoricalDatabase
from simulation.OrderbookSimulator import OrderbookSimulator
from gym.HistoricalOrderbookEnvironment import HistoricalOrderbookEnvironment
from rewards.RewardFunctions import InventoryAdjustedPnL, PnL

from features.Features import Portfolio

# import sns
from pylab import plt
import pandas as pd
import numpy as np
import os


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


def plot_per_episode(
        ticker,
        agent_name,
        step_size,
        market_order_clearing,
        reward_fun,
        step_info_per_episode,
        step_info_per_eval_episode,
        episode,
        done_info,
        done_info_eval
):
    step_info_per_episode = step_info_per_episode[episode]
    step_info_per_eval_episode = step_info_per_eval_episode[episode]

    def join_df(step_info_per_episode, step_info_per_eval_episode, train_metric, val_metric):
        train_metric = pd.DataFrame([train_metric], index=['training'],
                                    columns=step_info_per_episode.__dict__['dates'][-len(train_metric):]).T
        val_metric = pd.DataFrame([val_metric], index=['testing'],
                                  columns=step_info_per_eval_episode.__dict__['dates'][-len(val_metric):]).T
        metrics = pd.concat([train_metric, val_metric])
        assert(len(metrics.T)==2)
        assert(len(metrics)==(len(train_metric)+len(val_metric)))
        return train_metric, val_metric, metrics

    def graph_per_episode(step_info_per_episode, step_info_per_eval_episode, metric: str = None):
        train_metric = step_info_per_episode.__dict__[metric]
        val_metric = step_info_per_eval_episode.__dict__[metric]
        _, _, metrics = join_df(step_info_per_episode, step_info_per_eval_episode, train_metric, val_metric)
        return metrics

    def info_reward(step_info_per_episode, step_info_per_eval_episode, window: str = '10s'):
        train_reward = np.diff(step_info_per_episode.__dict__['pnls'])
        val_reward = np.diff(step_info_per_eval_episode.__dict__['pnls'])
        train_reward, test_reward, rewards = join_df(step_info_per_episode, step_info_per_eval_episode, train_reward, val_reward)
        reward_roll_mean = pd.concat([train_reward.sort_index().rolling(window).mean(), test_reward.sort_index().rolling(window).mean()])
        reward_roll_vol = pd.concat([train_reward.sort_index().rolling(window).std() ** 0.5, test_reward.sort_index().rolling(window).std() ** 0.5])
        stats = pd.DataFrame(rewards).describe()
        stats = np.round(stats)
        stats = stats.astype(int)
        return stats, rewards, reward_roll_mean, reward_roll_vol

    def info_actions(step_info_per_episode, step_info_per_eval_episode):
        actions_train = pd.DataFrame.from_dict(step_info_per_episode.__dict__['actions'], orient='index').T
        actions_test = pd.DataFrame.from_dict(step_info_per_eval_episode.__dict__['actions'], orient='index').T
        return actions_train.value_counts(), actions_test.value_counts()

    def info_factions(step_info_per_episode, step_info_per_eval_episode):
        actions_train = pd.DataFrame.from_dict(step_info_per_episode.__dict__['filled_actions'], orient='index').T
        factions_train = actions_train[actions_train!=-1][actions_train!=0].dropna()
        factions_train_ = actions_train[(actions_train == -1).sum(axis=1).astype('bool')]
        actions_test = pd.DataFrame.from_dict(step_info_per_eval_episode.__dict__['filled_actions'], orient='index').T
        factions_test = actions_test[actions_test!=-1][actions_test!=0].dropna()
        factions_test_ = actions_test[(actions_test == -1).sum(axis=1).astype('bool')]
        return factions_train.value_counts(), factions_train_.value_counts(), factions_test.value_counts(), factions_test_.value_counts()

    def list_of_df(metric: pd.DataFrame, window: str='10min'):
        return [v for k, v in metric.groupby(pd.Grouper(freq=window))]

    def uncertainties_pnl_map(step_info, window: str='10min'):
        cols = step_info.__dict__['dates'][1:]
        reward = pd.DataFrame(np.diff(step_info.__dict__['pnls']), index=cols)
        spread = pd.DataFrame(step_info.__dict__['spreads'][1:], index=cols)
        inventories = pd.DataFrame(step_info.__dict__['inventories'][1:], index=cols)
        l_df_rewards = list_of_df(reward, window)
        l_df_spread = list_of_df(spread, window)
        assert (len(l_df_rewards) == len(l_df_spread))
        l_df_inventories = list_of_df(inventories, window)
        l_nd_pnl = [np.sum(l_df_rewards[i]) / np.mean(l_df_spread[i]) for i in range(len(l_df_spread))]
        l_map = [np.mean(np.abs(l_df_inventories[i])) for i in range(len(l_df_inventories))]
        uncertainties = pd.DataFrame(index=['Normalised daily PnL', 'Mean Absolute Position'],
                                     columns=['Mean', 'Standard Deviation', 'Mean Absolute Deviation']).T
        uncertainties['Normalised daily PnL'] = [np.mean(l_nd_pnl), np.std(l_nd_pnl),
                                                 np.mean(np.absolute(l_nd_pnl - np.mean(l_nd_pnl)))]
        uncertainties['Mean Absolute Position'] = [np.mean(l_map), np.std(l_map),
                                                   np.mean(np.absolute(l_map - np.mean(l_map)))]
        return np.round(uncertainties, 1)

    def done_inf(dct):
        return np.round(pd.DataFrame.from_dict(dct, orient='index').T, 1)

    # sns.set()
    try:
        damp_factor = reward_fun.inventory_aversion
    except:
        damp_factor = None

    if damp_factor:
        reward_fun = 'Asymmetrically dampened PnL' if reward_fun.asymmetrically_dampened else 'Symmetrically dampened PnL'
    else:
        reward_fun = 'PnL'

    fig = plt.figure(constrained_layout=True, figsize=(10, 20))
    ax_dict = fig.subplot_mosaic(
        """
        ZY
        AB
        CD
        EF
        GH
        PQ
        IJ
        """
    )

    name = f"{ticker} - {agent_name} - Episode_{episode} \n step size: {step_size.total_seconds()}sec | " \
        + f"reward: {reward_fun} | dampening factor: {damp_factor}  | order clearing factor: {market_order_clearing}\n"

    plt.suptitle(name)

    done_info = done_inf(done_info)
    done_info_eval = done_inf(done_info_eval)

    table = ax_dict["Z"].table(
        cellText=done_info.values,
        colLabels=done_info.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    #table.scale(0.5, 1.1)
    ax_dict["Z"].set_axis_off()
    ax_dict["Z"].title.set_text("Agent's characteristics training")

    table = ax_dict["Y"].table(
        cellText=done_info_eval.values,
        colLabels=done_info_eval.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    #table.scale(0.5, 1.1)
    ax_dict["Y"].set_axis_off()
    ax_dict["Y"].title.set_text("Agent's characteristics testing")


    equity_curve = graph_per_episode(step_info_per_episode, step_info_per_eval_episode, 'pnls')
    equity_curve.plot(ax=ax_dict["A"], ylabel='aum', xlabel='n-th bar reaches',
                      title=f'Equity curve through time')

    inventory_curve = graph_per_episode(step_info_per_episode, step_info_per_eval_episode, 'inventories')
    inventory_curve.plot(ax=ax_dict["B"], ylabel='inventory', xlabel='n-th bar reaches',
                         title=f'Inventory curve through time')

    window = '30min'
    stats_rewards, rewards, rewards_roll_mean, rewards_roll_std = info_reward(step_info_per_episode,
                                                                              step_info_per_eval_episode,
                                                                              window)

    table = ax_dict["C"].table(
        cellText=stats_rewards.values,
        rowLabels=stats_rewards.index,
        colLabels=stats_rewards.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    #table.scale(0.5, 1.1)
    ax_dict["C"].set_axis_off()
    ax_dict["C"].title.set_text("Satistics of agent's reward (PnL)")

    non_null_reward = rewards[rewards != 0].dropna(how='all')
    non_null_reward.plot.hist(ax=ax_dict["D"], title="Non null rewards histogram", bins=50, alpha=0.5)

    if rewards_roll_mean is not None:
        rewards_roll_mean.plot(ax=ax_dict["E"], title=f'PnL rolling ({window}) mean')
        rewards_roll_std.plot(ax=ax_dict["F"], title=f'PnL rolling ({window}) volatility')

    actions_train, actions_test = info_actions(step_info_per_episode, step_info_per_eval_episode)
    actions_f2train, actions_f1train, actions_f2test, actions_f1test = info_factions(step_info_per_episode, step_info_per_eval_episode)
    actions_train, actions_test = actions_train.to_frame(
        name="Training"), actions_test.to_frame(name="Testing")
    actions_f2train, actions_f2test = actions_f2train.to_frame(
        name="Training"), actions_f2test.to_frame(name="Testing")
    actions_f1train, actions_f1test = actions_f1train.to_frame(
        name="Training"), actions_f1test.to_frame(name="Testing")
    actions = pd.concat([actions_train, actions_test], axis=1).sort_values(by='Training')
    actions_2filled = pd.concat([actions_f2train, actions_f2test], axis=1).sort_values(by='Training')
    actions_1filled = pd.concat([actions_f1train, actions_f1test], axis=1).sort_values(by='Training')

    actions.plot.barh(ax=ax_dict["G"], title="Count agent's parameter actions (bid, sell)")

    actions = actions.iloc[:-1] #delete market orders to count the pct of filled order, indeed market orders are always filled
    not_filled = (actions.sum().values - actions_2filled.sum().values - actions_1filled.sum().values) / actions.sum().values
    not_filled = np.round(pd.DataFrame(not_filled*100, index=['Training', 'Testing'], columns=['Not filled actions (%)']).T, 1)
    #not_filled.plot.barh(ax=ax_dict["P"], title = "Not filled actions in units")
    table = ax_dict["P"].table(
        cellText=not_filled.values,
        colLabels=not_filled.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    #table.scale(0.5, 1.1)
    ax_dict["P"].set_axis_off()
    ax_dict["P"].title.set_text("Agent's actions not filled in %")

    actions_2filled.plot.barh(ax=ax_dict["H"], title="Count agent's actions filled both side (bid, sell)")
    actions_1filled.plot.barh(ax=ax_dict["Q"], title="Count agent's actions filled one side (bid, sell)")

    window = '30min'
    uncertainties_train = uncertainties_pnl_map(step_info_per_episode, window)
    table = ax_dict["I"].table(
        cellText=uncertainties_train.values,
        rowLabels=uncertainties_train.index,
        colLabels=uncertainties_train.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    #table.scale(0.5, 1.1)
    ax_dict["I"].set_axis_off()
    ax_dict["I"].title.set_text(f"Training statistics of ND-PnL and MAP groupby {window}")

    uncertainties_test = uncertainties_pnl_map(step_info_per_eval_episode)
    table = ax_dict["J"].table(
        cellText=uncertainties_test.values,
        rowLabels=uncertainties_test.index,
        colLabels=uncertainties_test.columns,
        loc="center",
    )
    table.set_fontsize(6.5)
    #table.scale(0.5, 1.1)
    ax_dict["J"].set_axis_off()
    ax_dict["J"].title.set_text("Testing")


    pdf_path = os.path.join("results", agent_name)
    os.makedirs(pdf_path, exist_ok=True)
    subname = name.replace('\n', '').replace(' ', '_').replace(':', '').replace('|','')
    pdf_filename = os.path.join(pdf_path, f"Ep_{episode}_{subname}.pdf")
    # Write plot to pdf
    fig.savefig(pdf_filename)
    plt.close(fig)


def plot_final(
        done_info,
        done_info_eval,
        ticker,
        agent_name,
        step_size,
        market_order_clearing,
        reward_fun,
):
    def graph_final(done_info, done_info_eval, metric):
        metrics = pd.DataFrame([done_info[metric], done_info_eval[metric]], index=['training', 'testing']).T
        return metrics

    fig = plt.figure(constrained_layout=True, figsize=(15, 15))
    ax_dict = fig.subplot_mosaic(
        """
        AB
        CD
        """
    )

    try:
        damp_factor = reward_fun.inventory_aversion
    except:
        damp_factor = None
    if damp_factor:
        reward_fun = 'Asymmetrically dampened PnL' if reward_fun.asymmetrically_dampened else 'Symmetrically dampened PnL'
    else:
        reward_fun = 'PnL'
    name = f"{ticker} - {agent_name}\n step size in sec: {step_size.total_seconds()} | " \
        + f"reward: {reward_fun} | dampening factor: {damp_factor}  | market order clearing factor: {market_order_clearing}\n"
    pdf_path = os.path.join("results", agent_name)
    subname = name.replace('\n', '').replace(' ', '_').replace(':', '').replace('|', '')
    pdf_filename = os.path.join(pdf_path, f"final_{subname}.pdf")

    metrics = ['nd_pnl', 'map', 'aum', 'depth']
    for metric, ax in zip(metrics, ["A", "B", "C", "D"]):
        graph = graph_final(done_info, done_info_eval, metric)
        graph.plot(ax=ax_dict[ax], ylabel=metric, xlabel='n-th episodes', title=f'{metric} through episodes')

    fig.savefig(pdf_filename)
    plt.close(fig)