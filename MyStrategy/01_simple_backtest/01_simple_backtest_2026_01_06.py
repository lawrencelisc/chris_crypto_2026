import sys
import datetime

import numpy as np
import pandas as pd

from pathlib import Path

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def simple_backtest(df, candle_len, tp, sl, cycle):

    res = '4h'
    num_of_res = int(res[:-1])  # num of hours per cycle

    open_unix_datetime = 0
    open_price = 0
    pnl = 0
    asset = 0
    net = 0
    num_trade = 0

    for i, row in df.iterrows():
        now_unix_datetime = int(i.timestamp())

        diff_unix_datetime = now_unix_datetime - open_unix_datetime

        now_open = row['open']
        now_high = row['high']
        now_low = row['low']
        now_close = row['close']

        now_candle = now_close - now_open
        now_candle = round(now_candle, 2)

        trade_logic = now_candle > candle_len
        close_logic = diff_unix_datetime >= (cycle * num_of_res * 60 * 60)
        tp_cond = (now_close - open_price) >= tp
        sl_cond = (open_price - now_close) >= sl
        last_cond = i == df.index[-1]

        # ========== open position ==========
        if asset == 0 and trade_logic:
            asset = 1
            open_unix_datetime = now_unix_datetime
            open_price = now_close

        # ========== profit taking ==========
        elif asset > 0 and (tp_cond or sl_cond or last_cond or close_logic):
            asset = 0

            if sl_cond:
                pnl = -sl
            elif tp_cond:
                pnl = tp
            else:
                pnl = now_close - open_price

            net += pnl
            num_trade += 1

    print(f'candle_len: {candle_len} | tp: {tp} | sl: {sl} | cycle: {cycle}')
    print(f'net: {round(net, 2)} | num_trade: {num_trade}')

if __name__ == '__main__':

    # ========== read csv file ==========
    project_root = Path(__file__).parent.parent.parent
    crypto_data_path = project_root / 'crypto_data' / 'GN01_market_price_usd_ohlc_4h_BTC.csv'
    raw_df = pd.read_csv(crypto_data_path, parse_dates=['date'])
    raw_df = raw_df.set_index('date')

    # ========== initialization ==========
    dt_start = '2020-01-01 00:00:00'

    df = raw_df.copy()
    df = df.loc[dt_start:]

    candle_len_list = [600, 800, 1000]
    tp_list = [1500, 2000, 2500]
    sl_list = [300, 500, 800]
    cycle_list = [10, 15, 20]

    for candle_len in candle_len_list:
        for tp in tp_list:
            for sl in sl_list:
                for cycle in cycle_list:
                    simple_backtest(df, candle_len, tp, sl, cycle)