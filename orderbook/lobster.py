import pandas as pd
from os import path
import numpy as np
from itertools import chain
from datetime import timedelta

class LobsterData:
    """
    LOBSTER generates a 'message' and an 'orderbook' file for each active trading day of a selected ticker
    The 'orderbook' file contains the evolution of the limit order book up to the requested number of levels
    The 'message' file contains indicators for the type of event causing an update of the limit order book in the requested price range
    https://lobsterdata.com/info/DataStructure.php
    """
    def __init__(self, ticker: str = None):
        self.ticker = ticker
        self.order_books = pd.DataFrame()
        self.messages = pd.DataFrame()
        self.level = 0

    @property
    def events(self):
        return {1: 'submission',
                2: 'cancellation',
                3: 'deletion',
                4: 'execution',
                5: 'hidden_execution',
                6: 'cross_trade',
                7: 'trading_halt',
                8: 'other'}

    @property
    def directions(self):
        return {-1: 'sell',
                1: 'buy'}

    def read_date(self, order_book_file):
        """
        Find the trading day
        """
        file_name = path.basename(order_book_file)
        file_base = path.splitext(file_name)[0]
        return np.datetime64('{0}T00:00-0000'.format(file_base.split('_')[1]))

    def read_messages(self, mydate, message_file):
        """
        Read message of that trading day
        """
        messages = pd.read_csv(message_file, header=None)
        messages.columns = ['timestamp','event', 'id', 'size', 'price', 'direction']
        messages['price'] /= 10000
        messages['event'] = messages['event'].map(lambda i: self.events[i])
        messages['direction'] = messages['direction'].map(lambda i: self.directions[i])
        messages['timestamp'] = messages['timestamp'].apply(lambda x: mydate + np.timedelta64(int(x * 1e9), 'ns'))
        messages.set_index('timestamp', drop=True, inplace=True)
        messages['ticker'] = self.ticker
        return messages

    def read_orderbooks(self, order_book_file):
        orderbooks = pd.read_csv(order_book_file, header=None)
        n_levels = int(len(orderbooks.columns) / 4 + 0.5)
        price_cols = list(chain(*[(f"ask_price_{i},bid_price_{i}").split(",") for i in range(1,n_levels+1)]))
        volume_cols = list(chain(*[(f"ask_size_{i},buy_size_{i}").split(",") for i in range(1,n_levels+1)]))
        orderbooks.columns = list(chain(*zip(price_cols, volume_cols)))
        orderbooks[price_cols] /= 10000
        orderbooks['ticker'] = self.ticker
        return orderbooks

    def remove_hidden_exec(self, messages, orderbooks):
        new_index = messages['event'] != 'hidden_execution'
        messages = messages[new_index]
        orderbooks = orderbooks[new_index]
        return messages, orderbooks

    def read_daily_data(self, order_book_file, message_file, multi_days: bool = False):
        date = self.read_date(order_book_file)
        messages, orderbooks = self.read_messages(date, message_file), self.read_orderbooks(order_book_file)
        orderbooks.index = messages.index
        messages, orderbooks = self.remove_hidden_exec(messages, orderbooks)
        if multi_days:
            self.order_books = pd.concat([self.order_books, orderbooks], axis=0)
            self.messages = pd.concat([self.messages, messages], axis=0)
        else:
            self.order_books = orderbooks
            self.messages = messages
        self.level = int(self.order_books.shape[1] / 4 + 0.5)

    def read_period_data(self, level, start_date, end_date, data_path='./'):
        my_date = start_date
        while my_date <= end_date:
            print('read {0}'.format(my_date.strftime('%Y-%m-%d')))
            order_book_file = "{0}/{1}_{2}_34200000_57600000_orderbook_{3}.csv".format(data_path, self.ticker,
                                                                                       my_date.strftime('%Y-%m-%d'),
                                                                                       level)

            message_file = "{0}/{1}_{2}_34200000_57600000_message_{3}.csv".format(data_path, self.ticker,
                                                                                       my_date.strftime('%Y-%m-%d'),
                                                                                       level)
            self.read_daily_data(order_book_file, message_file, True)
            my_date += timedelta(1)

    def get_number_of_record(self):
        return self.order_books.shape[0]