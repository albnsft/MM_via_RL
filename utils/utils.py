from datetime import datetime, timedelta
from orderbook.lobster import LobsterData


def get_date_time(date_string: str):
    return datetime.strptime(date_string, "%Y-%m-%d")


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def convert_timedelta_to_freq(delta: timedelta):
    assert sum([delta.seconds > 0, delta.microseconds > 0]) == 1, "Timedelta must be given in seconds or microseconds."
    if delta.seconds > 0:
        return f"{delta.seconds}S"
    else:
        return f"{delta.microseconds}ms"


def read_data(ticker: str = None):
    book_file = f'data/{ticker}_2012-06-21_34200000_57600000_orderbook_5.csv'
    message_file = f'data/{ticker}_2012-06-21_34200000_57600000_message_5.csv'
    lobster = LobsterData(ticker)
    lobster.read_daily_data(book_file, message_file)
    return lobster.messages, lobster.order_books


def split_dates(split: float = None, date: datetime = None, hour_start: float = None, hour_end: float = None,
                step_in_sec: float = None):
    start_train_date = date + timedelta(hours=hour_start) + timedelta(seconds=step_in_sec)
    end_test_date = date + timedelta(hours=hour_end)
    end_train_date = (start_train_date + (end_test_date - start_train_date) * split).replace(microsecond=0)
    start_test_date = end_train_date + timedelta(seconds=step_in_sec)
    return [start_train_date, end_train_date, start_test_date, end_test_date]
