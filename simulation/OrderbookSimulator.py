import sys
from collections import deque
from datetime import datetime, timedelta

if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    from typing import Deque, Dict, List, Optional, Literal, Callable, cast
else:
    from typing import Deque, Dict, List, Optional, Callable
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from database.HistoricalDatabase import HistoricalDatabase
from orderbook.models import Orderbook, Order, LimitOrder, FilledOrders, OrderDict
from orderbook.Exchange import Exchange
from simulation.HistoricalOrderGenerator import HistoricalOrderGenerator


class OrderbookSimulator:
    def __init__(
        self,
        ticker: str = "MSFT",
        exchange: Exchange = None,
        order_generator: HistoricalOrderGenerator = None,
        n_levels: int = 5,
        database: HistoricalDatabase = None,
        outer_levels: int = 5,
        trading_date: datetime = datetime(2012, 6, 21),
        verbose: bool = False
    ) -> None:
        self.ticker = ticker
        self.exchange = exchange or Exchange(ticker)
        self.order_generator = order_generator or HistoricalOrderGenerator(ticker, database)
        self.now_is: datetime = datetime(2000, 1, 1)
        self.trading_date = trading_date
        self.n_levels = n_levels
        self.database = database or HistoricalDatabase()
        self.outer_levels = outer_levels
        self.verbose = verbose
        # The following is for re-syncronisation with the historical data
        self.max_sell_price: int = 0
        self.min_buy_price: int = np.infty  # type:ignore
        self.initial_buy_price_range: int = np.infty  # type:ignore
        self.initial_sell_price_range: int = np.infty  # type:ignore

    def reset_episode(self, start_date: datetime, start_book: Optional[Orderbook] = None):
        if not start_book:
            start_book = self.get_historical_start_book(start_date)
        self.exchange.central_orderbook = start_book
        self._reset_initial_price_ranges()
        assert start_date.microsecond == 0, "Episodes must be started on the second."
        self.now_is = start_date
        return start_book

    def forward_step(self, until: datetime, internal_orders: Optional[List[Order]] = None) -> FilledOrders:
        assert (
            until > self.now_is
        ), f"The current time is {self.now_is.time()}, but we are trying to step forward in time until {until.time()}!"
        external_orders = list(self.order_generator.generate_orders(self.now_is, until))
        orders = internal_orders or list()
        orders += external_orders
        filled_internal_orders = []
        filled_external_orders = []
        for order in orders:
            filled = self.exchange.process_order(order)
            if filled:
                filled_internal_orders += filled.internal
                filled_external_orders += filled.external
        self.now_is = until
        if (self._near_exiting_initial_price_range or self._exiting_worst_price) :
            self.update_outer_levels()
        return FilledOrders(internal=filled_internal_orders, external=filled_external_orders)

    def get_historical_start_book(self, start_date: datetime) -> Orderbook:
        start_series = self.database.get_last_snapshot(start_date, ticker=self.ticker)
        assert len(start_series) > 0, f"There is no data before the episode start time: {start_date}"
        assert start_date - start_series.name <= timedelta(
            days=1
        ), f"Attempting to get data from > a day ago (start_date: {start_date}; start_series.name: {start_series.name})"
        initial_orders = self._get_initial_orders_from_snapshot(start_series)
        return self.exchange.get_initial_orderbook_from_orders(initial_orders)

    def _initial_prices_filter_function(self, direction: Literal["buy", "ask"], price: int) -> bool:
        if direction == "buy" and price < self.min_buy_price or direction == "sell" and price > self.max_sell_price:
            return True
        else:
            return False

    def update_outer_levels(self) -> None:
        """Update levels that had no initial orders at the start of the episode. If this is not done, the volume at
        these levels will be substantially lower than it should be. If agent orders exist at these levels, they will be
        cancelled and replaced at the back of the queue."""
        if self.verbose: print(f"Updating outer levels. Current time is {self.now_is}.")
        orderbook_series = self.database.get_last_snapshot(self.now_is, ticker=self.ticker)
        orders_to_add = self._get_initial_orders_from_snapshot(orderbook_series, self._initial_prices_filter_function)
        for order in orders_to_add:
            """
            try:
                to_delete = getattr(self.exchange.central_orderbook, order.direction)[order.price]
                type_deleted = [getattr(delet, 'is_external') for delet in to_delete]
                if 'False' in type_deleted: print('an internal order has been replaced by externals, seems wrong')
            except KeyError:
                pass
            """
            getattr(self.exchange.central_orderbook, order.direction)[order.price] = deque([order])
        self.min_buy_price = min(self.min_buy_price, self.exchange.orderbook_price_range[0])
        self.max_sell_price = max(self.max_sell_price, self.exchange.orderbook_price_range[1])

    @staticmethod
    def _remove_hidden_executions(messages: pd.DataFrame):
        return messages[messages.message_type != "execution_hidden"]

    always_true_function: Callable = lambda direction, price: True

    def _get_initial_orders_from_snapshot(self, series: pd.DataFrame, filter_function: Callable = always_true_function):
        initial_orders = []
        for direction in ["buy", "sell"]:
            for level in range(self.n_levels):
                if f"{direction}_volume_{level}" not in series:
                    continue
                if filter_function(direction, series[f"{direction}_price_{level}"]):
                    initial_orders.append(
                        LimitOrder(
                            timestamp=series.name,
                            price=series[f"{direction}_price_{level}"],
                            volume=series[f"{direction}_volume_{level}"],
                            direction=direction,  # type: ignore
                            ticker=self.ticker,
                            internal_id=-1,
                            external_id=None,
                            is_external=True,
                        )
                    )
        return initial_orders

    @property
    def _near_exiting_initial_price_range(self) -> bool:
        outer_proportion = self.outer_levels / self.n_levels
        return (
            self.exchange.best_buy_price < self.min_buy_price + outer_proportion * self.initial_buy_price_range
            or self.exchange.best_sell_price > self.max_sell_price - outer_proportion * self.initial_sell_price_range
        )

    @property
    def _exiting_worst_price(self) -> bool:
        worst_buy, worst_sell = self.exchange.orderbook_price_range
        return (
            worst_buy < self.min_buy_price
            or worst_sell > self.max_sell_price
        )

    def _reset_initial_price_ranges(self):
        self.min_buy_price, self.max_sell_price = self.exchange.orderbook_price_range
        self.initial_buy_price_range = self.exchange.best_buy_price - self.min_buy_price
        self.initial_sell_price_range = self.max_sell_price - self.exchange.best_sell_price

    """
     internal_book_side = getattr(self.exchange.internal_orderbook, order.direction)
     internal_orders_to_cancel = list()
     if order.price in internal_book_side.keys():
         if self.verbose: print(
             "Resynchronising levels containing internal orders. These internal orders will be cancelled and"
             + " replaced at the back of the queue."
         )
         for internal_order in internal_book_side[order.price]:
             order_dict = cast(OrderDict, internal_order.__dict__)
             order_dict["timestamp"] = self.now_is
             cancellation = create_order("cancellation", order_dict)
             limit = create_order("limit", order_dict)
             internal_orders_to_cancel.append(cancellation)
             internal_orders_to_replace.append(limit)
     for cancellation in internal_orders_to_cancel:
         self.exchange.process_order(cancellation)
     """

    """
    def add_levels(self, filled_order) -> None:
        orderbook_series = self.database.get_last_snapshot(filled_order.timestamp, ticker=self.ticker)
        opposite_direction = "sell" if filled_order.direction == "buy" else "buy"
        orders = [order for order in self._get_initial_orders_from_snapshot(orderbook_series) if order.direction == opposite_direction]
        for order in orders:
            if not order.price in getattr(self.exchange.central_orderbook, opposite_direction).keys():
                getattr(self.exchange.central_orderbook, opposite_direction)[order.price] = deque([order])
        self.min_buy_price = min(self.min_buy_price, self.exchange.orderbook_price_range[0])
        self.max_sell_price = max(self.max_sell_price, self.exchange.orderbook_price_range[1])
    """

    """
    if len(self.exchange.central_orderbook.buy) < self.n_levels or \
        len(self.exchange.central_orderbook.sell) < self.n_levels:
    print(order.timestamp)
    self.add_levels(order)
    try:
        assert(len(self.exchange.central_orderbook.buy) == self.n_levels)
        assert (len(self.exchange.central_orderbook.sell) == self.n_levels)
    except:
        print('not same levels')
    """