from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from orderbook.lobster import LobsterData

from database.HistoricalDatabase import HistoricalDatabase


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


def save_best_checkpoint_path(path_to_save_dir: str, best_checkpoint_path: str):
    text_file = open(path_to_save_dir + "/best_checkpoint_path.txt", "wt")
    text_file.write(best_checkpoint_path)
    text_file.close()


def get_last_trading_dt(timestamp: datetime):
    ncal = mcal.get_calendar("NASDAQ")
    last_trading_date = ncal.schedule(start_date=timestamp - timedelta(days=4), end_date=timestamp).iloc[-1, 0].date()
    return datetime.combine(last_trading_date, datetime.min.time()) + timedelta(hours=16)


def get_next_trading_dt(timestamp: datetime):
    ncal = mcal.get_calendar("NASDAQ")
    next_trading_date = ncal.schedule(start_date=timestamp, end_date=timestamp + timedelta(days=4)).iloc[0, 0].date()
    return datetime.combine(next_trading_date, datetime.min.time()) + timedelta(hours=9, minutes=30)


def get_trading_datetimes(start_date: datetime, end_date: datetime):
    ncal = mcal.get_calendar("NASDAQ")
    return ncal.schedule(start_date=start_date, end_date=end_date).market_open.index


def daterange_in_db(start: datetime, end: datetime, ticker: str):
    database = HistoricalDatabase()
    next_snapshot = database.get_next_snapshot(start, ticker)
    last_snapshot = database.get_last_snapshot(end, ticker)
    if len(next_snapshot) == 0 or len(last_snapshot) == 0:
        return False
    bool_1 = next_snapshot.name - start < timedelta(minutes=1)
    bool_2 = end - last_snapshot.name < timedelta(minutes=1)
    return bool_1 and bool_2


def get_timedelta_from_clock_time(clock_time: int = 1000):
    t = datetime.strptime(str(clock_time), "%H%M")
    return timedelta(hours=t.hour, minutes=t.minute)


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
    end_train_date = start_train_date + (end_test_date - start_train_date) * split
    start_test_date = end_train_date + timedelta(seconds=step_in_sec)
    return [start_train_date, end_train_date, start_test_date, end_test_date]
