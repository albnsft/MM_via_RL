from datetime import datetime
from typing import List, Union

import pandas as pd

import os
from database.database_population_helpers import (
    get_book_snapshots,
    get_file_len,
    get_book_and_message_columns,
    get_book_and_message_paths,
    reformat_message_data,
)


class HistoricalDatabase:
    def __init__(self):
        self.exchange = "NASDAQ"  # This database currently only retrieves NASDAQ (LOBSTER) data
        self.init()

    def init(self, ticker: str = "MSFT"):
        n_levels = 5
        book_snapshot_freq = "S"
        path_to_lobster_data: str = os.path.abspath(__file__).replace('\database\HistoricalDatabase.py', "\data")
        trading_date = '2012-06-21'
        book_cols, message_cols = get_book_and_message_columns(n_levels)
        book_path, message_path = get_book_and_message_paths(path_to_lobster_data, ticker, trading_date, n_levels)
        n_messages = get_file_len(message_path)
        messages = pd.read_csv(message_path, header=None,names=message_cols)
        self.messages = reformat_message_data(messages, trading_date, ticker)
        self.books = get_book_snapshots(book_path, book_cols, messages, book_snapshot_freq, n_levels, n_messages)
        self.messages.set_index(['timestamp'], drop=False, inplace=True)

    def get_last_snapshot(self, timestamp: datetime, ticker: str):
        return self.books.loc[self.books.index <= timestamp].iloc[-1]

    def get_messages(self, start_date: datetime, end_date: datetime, ticker: str):
        messages = self.messages.loc[(self.messages.index > start_date) & (self.messages.index <= end_date)]
        if len(messages) > 0:
            return messages
        else:
            return pd.DataFrame()
