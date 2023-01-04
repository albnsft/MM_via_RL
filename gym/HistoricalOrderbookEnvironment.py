from __future__ import annotations

import warnings
from copy import deepcopy
from datetime import datetime, timedelta
import sys

from gym.order_tracking.InfoCalculators import InfoCalculator

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import List, Literal, Optional
else:
    from typing import List
    from typing_extensions import Literal

import numpy as np

from features.Features import (
    Feature,
    Spread,
    State,
    PriceMove,
    Volatility,
    Inventory,
    Portfolio,
    BookImbalance,
    TradeDirectionImbalance,
    TradeVolumeImbalance,
    RSI,
    BuyDistance,
    SellDistance
)
from gym.action_interpretation.OrderDistributors import OrderDistributor
from orderbook.create_order import create_order
from orderbook.models import Order, Orderbook, OrderDict, Cancellation, FilledOrders, MarketOrder
from rewards.RewardFunctions import RewardFunction, InventoryAdjustedPnL
from simulation.OrderbookSimulator import OrderbookSimulator


class HistoricalOrderbookEnvironment:
    def __init__(
            self,
            features: List[Feature] = None,
            ticker: str = "MSFT",
            step_size: timedelta = timedelta(seconds=0.1),
            initial_portfolio: Portfolio = None,
            start_of_trading: datetime = datetime(2012, 6, 21, 9, 30) + timedelta(seconds=1),
            end_of_trading: datetime = datetime(2012, 6, 21, 16),
            simulator: OrderbookSimulator = None,
            market_order_clearing: bool = True,
            market_order_fraction_of_inventory: float = 0.5,
            max_inventory: int = 1000,
            per_step_reward_function: RewardFunction = InventoryAdjustedPnL(inventory_aversion=0.1),
            info_calculator: InfoCalculator = None,
            order_distributor: OrderDistributor = None,
            normalisation_on: bool = True,
            n_levels: int = 5,
            n_lags_feature: int = 0,
            verbose: bool = False
    ):
        super(HistoricalOrderbookEnvironment, self).__init__()

        self.ticker = ticker
        self.step_size = step_size
        self.start_of_trading = start_of_trading
        self.end_of_trading = end_of_trading
        self.initial_portfolio = initial_portfolio or Portfolio(inventory=0, cash=0, gain=0)
        self.order_distributor = order_distributor or OrderDistributor()
        self.market_order_clearing = market_order_clearing
        self.market_order_fraction_of_inventory = market_order_fraction_of_inventory
        self.per_step_reward_function_midprice = per_step_reward_function
        self.terminal_reward_function = per_step_reward_function
        self.n_levels = n_levels
        self.n_lags_feature = n_lags_feature if n_lags_feature==0 else n_lags_feature-1
        self.info_calculator = info_calculator or InfoCalculator(verbose=verbose)
        self.pricer = lambda orderbook: orderbook.midprice
        self._check_params()
        self.max_inventory = max_inventory
        self.verbose = verbose
        self.features = features or self.get_default_features(step_size, normalisation_on)
        self.max_feature_window_size = max([feature.window_size for feature in self.features])
        self.simulator = simulator or OrderbookSimulator(
            ticker=ticker,
            n_levels=self.n_levels,
            verbose=verbose
        )
        self.state: State = self._get_default_state()
        self.date_threshold: datetime = self.start_of_trading + (self.end_of_trading - self.start_of_trading) * 1

    def reset(self) -> np.ndarray:
        now_is = self.start_of_trading
        self.simulator.reset_episode(start_date=now_is)
        price = self.pricer(self.central_orderbook)
        self.state = State(FilledOrders(), self.central_orderbook, price, self.initial_portfolio, now_is, None, None)
        self._reset_features(now_is)
        if self.n_lags_feature > 0: self.lags_feature = np.zeros((self.n_lags_feature+1, len(self.features)))
        for step in range(int(self.max_feature_window_size / self.step_size) + self.n_lags_feature):
            self._forward(list())
            self._update_features()
            if self.n_lags_feature > 0:
                self._set_lags_features(step)
        return self.get_features()

    def step(self, action: int):
        done = False
        internal_orders = self.convert_action_to_orders(action=action)
        current_state = deepcopy(self.state)
        self._forward(internal_orders)
        self._update_features()
        next_state = self.state
        reward_relative_midprice = self.per_step_reward_function_midprice.calculate(current_state, next_state)
        features = self.get_features()
        if self.end_of_trading <= next_state.now_is or (self.mark_to_market_value < -1000 and next_state.now_is > self.date_threshold):
            reward_relative_midprice = self.terminal_reward_function.calculate(current_state, next_state)
            done = True
        info = self.info_calculator.calculate(self.state, reward_relative_midprice)
        return features, reward_relative_midprice, done, info

    def _forward(self, internal_orders: List[Order]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filled = self.simulator.forward_step(until=self.state.now_is + self.step_size, internal_orders=internal_orders)
        self.update_internal_state(filled)
        return filled

    def _get_features(self) -> np.ndarray:
        return np.array([feature.current_value for feature in self.features])

    def get_features(self) -> np.ndarray:
        if self.n_lags_feature == 0:
            return self._get_features()
        else:
            self.lags_feature[-1] = self._get_features()
            return self.lags_feature

    def update_internal_state(self, filled_orders: FilledOrders):
        self._update_portfolio(filled_orders)
        self.state.filled_orders = filled_orders
        self.state.orderbook = self.central_orderbook
        self.state.price = self.pricer(self.central_orderbook)
        self.state.now_is += self.step_size

    def convert_action_to_orders(self, action: int) -> List[Order]:
        tetha_sell, tetha_buy, prices = self.order_distributor.convert_action(action, self.state.orderbook)
        if self.market_order_clearing and np.abs(
                self.state.portfolio.inventory) > self.max_inventory:  # cancel all orders
            orders = self._get_inventory_clearing_market_order()
            vol = sum(order.volume for order in orders)
            if self.verbose:
                print(f'{self.state.portfolio.inventory} current inventory')
                print(f'{orders[0].timestamp} Market order clearing: {orders[0].direction} for a volume of {vol}')
            self.state.buy_parameter = 0
            self.state.sell_parameter = 0
        else:
            orders = self._get_limit_orders(prices, self.order_distributor.volume)
            self.state.buy_parameter = tetha_buy
            self.state.sell_parameter = tetha_sell
        return orders

    def _reset_features(self, episode_start: datetime):
        for feature in self.features:
            first_usage_time = episode_start - feature.window_size
            feature.reset(self.state, first_usage_time)

    def _update_features(self):
        for feature in self.features:
            feature.update(self.state)

    def _get_limit_orders(self, prices: dict[str, float], order_volume: int) -> List[Order]:
        orders = list()
        for side in ["buy", "sell"]:
            order_dict = self._get_default_order_dict(side)  # type:ignore
            order_dict["price"] = prices[side]
            order_dict["volume"] = order_volume
            order = create_order("limit", order_dict)
            orders.append(order)
        return orders

    def _get_inventory_clearing_market_order(self) -> List[MarketOrder]:
        inventory = self.state.portfolio.inventory
        order_direction = "buy" if inventory < 0 else "sell"
        order_dict = self._get_default_order_dict(order_direction)  # type:ignore
        order_dict["volume"] = np.round(np.abs(inventory) * self.market_order_fraction_of_inventory)
        market_order = create_order("market", order_dict)
        return [market_order]

    def _update_portfolio(self, filled_orders: FilledOrders):
        self.state.portfolio.gain = 0
        for order in filled_orders.internal:
            if order.direction == "sell":
                self.state.portfolio.inventory -= order.volume
                self.state.portfolio.cash += order.volume * order.price
                self.state.portfolio.gain += order.volume * (order.price - self.state.price)
            elif order.direction == "buy":
                self.state.portfolio.inventory += order.volume
                self.state.portfolio.cash -= order.volume * order.price
                self.state.portfolio.gain += order.volume * (self.state.price - order.price)
        """
        from orderbook.models import LimitOrder, MarketOrder
        if len(filled_orders.internal)>0:
            print('*' * 50)
            dir = [order.direction for order in filled_orders.internal]
            print('Executed internal limit orders')
            print(dir)
            if isinstance(filled_orders.internal[0], LimitOrder): print(f'{filled_orders.internal[0].timestamp} gain for limit orders: {self.state.portfolio.gain}')
            if isinstance(filled_orders.internal[0], MarketOrder): print(f'{filled_orders.internal[0].timestamp} gain for market orders: {self.state.portfolio.gain}')
        """

    @property
    def central_orderbook(self):
        return self.simulator.exchange.central_orderbook

    @property
    def internal_orderbook(self):
        return self.simulator.exchange.internal_orderbook

    @property
    def mark_to_market_value(self):
        return self.state.portfolio.inventory * self.state.price + self.state.portfolio.cash

    def _check_params(self):
        assert self.start_of_trading <= self.end_of_trading, "Start of trading Nonsense"

    def _set_lags_features(self, step):
        if step == int(self.max_feature_window_size / self.step_size) - 1:
            self.lags_feature[0] = self._get_features()
        elif step >= int(self.max_feature_window_size / self.step_size):
            self.lags_feature[step - int(self.max_feature_window_size / self.step_size) + 1] = self._get_features()

    def _get_default_order_dict(self, direction: Literal["buy", "sell"]) -> OrderDict:
        return OrderDict(
            timestamp=self.state.now_is,
            price=None,
            volume=None,
            direction=direction,
            ticker=self.ticker,
            internal_id=None,
            external_id=None,
            is_external=False,
        )

    def _get_default_state(self):
        return State(
            filled_orders=FilledOrders(),
            orderbook=self.simulator.exchange.get_empty_orderbook(),
            price=0.0,
            portfolio=self.initial_portfolio,
            now_is=datetime.min,
            buy_parameter=None,
            sell_parameter=None
        )

    @staticmethod
    def get_default_features(step_size: timedelta, normalisation_on: bool = False):
        if step_size > timedelta(seconds=0.1): step_size = timedelta(seconds=0.1)
        return [
            Spread(
                update_frequency=step_size,
                normalisation_on=normalisation_on
            ),
            BookImbalance(update_frequency=step_size,
                          normalisation_on=normalisation_on
                          ),
            PriceMove(
                name="price_move_0.1_s",
                update_frequency=step_size,
                lookback_periods=1,
                normalisation_on=normalisation_on,
            ),
            PriceMove(
                name="price_move_1_s",
                update_frequency=step_size,
                lookback_periods=10,
                normalisation_on=normalisation_on,
            ),
            Volatility(
                name="volatility_6_s",
                update_frequency=step_size,
                lookback_periods=60,
                normalisation_on=normalisation_on,
            ),
            Volatility(
                name="volatility_12_s",
                update_frequency=step_size,
                lookback_periods=int(2 * 60),
                normalisation_on=normalisation_on,
            ),
            RSI(
                name="rsi_6_s",
                update_frequency=step_size,
                lookback_periods=int(60),
                normalisation_on=normalisation_on,
            ),
            RSI(
                name="rsi_12_s",
                update_frequency=step_size,
                lookback_periods=int(2 * 60),
                normalisation_on=normalisation_on,
            ),
            TradeVolumeImbalance(
                update_frequency=step_size,
                lookback_periods=int(60),
                normalisation_on=normalisation_on,
            ),
            TradeDirectionImbalance(
                update_frequency=step_size,
                lookback_periods=int(60),
                normalisation_on=normalisation_on,
            ),
            Inventory(
                update_frequency=step_size,
                normalisation_on=normalisation_on
            ),
            BuyDistance(
                update_frequency=step_size,
                normalisation_on=normalisation_on
            ),
            SellDistance(
                update_frequency=step_size,
                normalisation_on=normalisation_on
            )

        ]
