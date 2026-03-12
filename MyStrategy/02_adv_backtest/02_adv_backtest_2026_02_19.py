import sys
import datetime

import numpy as np
import pandas as pd

from pathlib import Path

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def backtest(df, initial_capital, candle_len,
             candle_dir, tp, sl, cycle, sma_dir,
             sma_len, std_ratio_thres):

    df['sma'] = df['close'].rolling(sma_len).mean()
    df['std'] = df['close'].rolling(sma_len).std()
    df['std_raito'] = (df['sma'] - df['close']) / df['std']

    # ========== initialization (function) ==========

    open_t = datetime.datetime.now().date()
    last_realized_capital = initial_capital
    res = '4h'
    num_of_res = int(res[:-1])  # num of hours per cycle

    open_price = 0
    num_of_coin = 0
    net_profit = 0
    num_of_trade = 0

    equity_value = 0
    realized_pnl = 0
    unrealized_pnl = 0

    commission_rate = 0.0055 / 100

    for i, row in df.iterrows():
        now_t = i.date()
        now_open = row['open']
        now_high = row['high']
        now_low = row['low']
        now_close = row['close']

        now_candle = round(now_close - now_open, 2)
        now_sma = row['sma']
        now_std_raito = row['std_raito']

        ##### commission #####
        if num_of_coin > 0:
            open_fee = (num_of_coin * open_price) * commission_rate
            close_fee = (num_of_coin * now_close) * commission_rate
            commission = open_fee + close_fee
        else:
            commission = 0

        ##### equity value #####
        unrealized_pnl = num_of_coin * (now_close - open_price) - commission
        equity_value   = last_realized_capital + unrealized_pnl
        net_profit     = round(equity_value - initial_capital, 2)

        if candle_dir == 'positive':
            trade_logic = now_candle > candle_len * 0.01
        elif candle_dir == 'negative':
            trade_logic = now_candle < -1 * candle_len * 0.01

        if sma_dir == 'above':
            trade_logic = trade_logic and (now_close > now_sma) and now_std_raito < -1 * std_ratio_thres
        elif sma_dir == 'below':
            trade_logic = trade_logic and (now_close < now_sma) and now_std_raito > std_ratio_thres

        t_diff_hr = (now_t - open_t).total_seconds() / 3600
        close_logic = (t_diff_hr / num_of_res) >= cycle
        tp_cond = open_price != 0 and (now_close - open_price > tp * 0.01 * open_price)
        sl_cond = open_price != 0 and (open_price - now_close > sl * 0.01 * open_price)
        last_index_cond = i == df.index[-1]

        ##### open position #####
        if num_of_coin == 0 and not last_index_cond and trade_logic:

            num_of_coin = last_realized_capital / now_close

            open_price = now_close
            open_t = now_t

        ##### close position #####
        elif num_of_coin > 0 and (tp_cond or sl_cond or last_index_cond or close_logic):

            realized_pnl = unrealized_pnl
            last_realized_capital += realized_pnl

            num_of_trade += 1
            num_of_coin = 0

    return net_profit, num_of_trade


if __name__ == '__main__':

    # ========== read csv file ==========
    project_root = Path(__file__).parent.parent.parent
    crypto_data_path = project_root / 'crypto_data' / 'GN01_market_price_usd_ohlc_4h_BTC.csv'
    raw_df = pd.read_csv(crypto_data_path, parse_dates=['date'])
    raw_df = raw_df.set_index('date')

    # ========== initialization ==========
    dt_start = '2024-01-01 00:00:00'
    initial_capital = 10000

    df = raw_df.copy()
    df = df.loc[dt_start:]

    candle_dir_list = ['positive', 'negative']
    candle_len_list = [1, 2, 4]
    sma_len_list = [10, 15, 20]
    sma_dir_list = ['above', 'below', 'whatever']
    std_ratio_thres_list = [1.0, 2.0, 2.5]
    tp_list = [3, 5, 7, 10]
    sl_list = [1, 2]
    cycle_list = [10, 15, 20]

    result_dict = {}
    result_dict['net_profit'] = []
    result_dict['num_of_trade'] = []
    result_dict['std_ratio_thres'] = []
    result_dict['sma_len'] = []
    result_dict['sma_dir'] = []
    result_dict['candle_dir'] = []
    result_dict['cycle'] = []
    result_dict['sl'] = []
    result_dict['tp'] = []
    result_dict['candle_len'] = []

    for std_ratio_thres in std_ratio_thres_list:
        for sma_len in sma_len_list:
            for sma_dir in sma_dir_list:
                for candle_dir in candle_dir_list:
                    for cycle in cycle_list:
                        for sl in sl_list:
                            for tp in tp_list:
                                for candle_len in candle_len_list:
                                    net_profit, num_of_trade = backtest(df,
                                                                        initial_capital, candle_len,
                                                                        candle_dir, tp, sl, cycle, sma_dir,
                                                                        sma_len, std_ratio_thres)

                                    print('net_profit:', net_profit)
                                    print('num_of_trade:', num_of_trade)
                                    print('std_ratio_threshold:', std_ratio_thres)
                                    print('sma_len:', sma_len)
                                    print('candle_len:', candle_len)
                                    print('sma_direction:', sma_dir)
                                    print('candle_direction:', candle_dir)
                                    print('holding_day:', cycle)
                                    print('profit_target:', tp)
                                    print('stop_loss:', sl)
                                    print('---------------------------')

                                    result_dict['net_profit'].append(net_profit)
                                    result_dict['num_of_trade'].append(num_of_trade)
                                    result_dict['std_ratio_thres'].append(std_ratio_thres)
                                    result_dict['sma_len'].append(sma_len)
                                    result_dict['sma_dir'].append(sma_dir)
                                    result_dict['candle_dir'].append(candle_dir)
                                    result_dict['cycle'].append(cycle)
                                    result_dict['sl'].append(sl)
                                    result_dict['tp'].append(tp)
                                    result_dict['candle_len'].append(candle_len)

    result_df = pd.DataFrame(result_dict)
    result_df = result_df.sort_values(by='net_profit', ascending=False)
    print(result_df)