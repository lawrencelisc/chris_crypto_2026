import os
import sys
import time
from datetime import datetime
from pathlib import Path

import hkfdb

import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import multiprocessing as mp

import plotguy
import itertools


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def backtest(para_comb):

    para_dict       = para_comb['para_dict']
    sec_profile     = para_comb['sec_profile']
    reference_index = para_comb['reference_index']
    start_date      = para_comb['start_date']
    end_date        = para_comb['end_date']
    freq            = para_comb['freq']
    output_folder   = para_comb['output_folder']
    file_format     = para_comb['file_format']
    df              = para_comb['df']
    intraday        = para_comb['intraday']
    summary_mode    = para_comb['summary_mode']
    py_filename     = para_comb['py_filename']

    # ========== strategy specific ==========
    code            = para_comb['code']
    tp              = para_comb['tp']
    sl              = para_comb['sl']
    cycle           = para_comb['cycle']

    rsi_chg_dir     = para_comb['rsi_chg_dir']
    rsi_chg_thres   = para_comb['rsi_chg_thres']

    # ========== secondary profile ==========
    market          = sec_profile['market']
    sectype         = sec_profile['sectype']
    initial_capital = sec_profile['initial_capital']
    commission_rate = sec_profile['commission_rate']

    df = df.copy()

    # ========== strategy specific ==========

    if rsi_chg_dir == 'positive': df['trade_logic'] = df['rsi_change'] > rsi_chg_thres
    if rsi_chg_dir == 'negative': df['trade_logic'] = df['rsi_change'] <= -1 * rsi_chg_thres

    # ========== initialization (function) ==========

    df['action'] = ''
    df['num_of_share'] = 0

    df['open_price'] = np.NaN
    df['close_price'] = np.NaN

    df['realized_pnl'] = np.NaN
    df['unrealized_pnl'] = 0
    df['net_profit'] = 0

    df['equity_value'] = initial_capital
    df['mdd_dollar'] = 0
    df['mdd_pct'] = 0

    df['commission'] = 0
    df['logic'] = None

    open_t = datetime.now().date()
    open_price = 0
    num_of_coin = 0
    net_profit = 0
    num_of_trade = 0

    last_realized_capital = initial_capital

    equity_value = 0
    realized_pnl = 0
    unrealized_pnl = 0

    for i, row in df.iterrows():
        now_t = pd.to_datetime(i).date()
        now_open = row['open']
        now_high = row['high']
        now_low = row['low']
        now_close = row['close']

        # ========== strategy specific ==========

        trade_logic = row['trade_logic']

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

        if trade_logic: df.at[i, 'logic'] = 'trade_logic'

        t_diff_hr = (now_t - open_t).total_seconds() / 3600
        if (freq == '1D'): num_of_res = 24
        close_logic = (t_diff_hr / num_of_res) >= cycle
        tp_cond = open_price != 0 and (now_close - open_price > tp * 0.01 * open_price)
        sl_cond = open_price != 0 and (open_price - now_close) > sl * 0.01 * open_price
        last_index_cond = i == df.index[-1]

        ##### open position #####
        if num_of_coin == 0 and not last_index_cond and trade_logic:

            num_of_coin = last_realized_capital / now_close

            open_price = now_close
            open_t = now_t

            df.at[i, 'action'] = 'open'
            df.at[i, 'open_price'] = open_price

        ##### close position #####
        elif num_of_coin > 0 and (tp_cond or sl_cond or last_index_cond or close_logic):

            realized_pnl = unrealized_pnl
            unrealized_pnl = 0
            last_realized_capital += realized_pnl

            num_of_trade += 1
            num_of_coin = 0

            if close_logic: df.at[i, 'logic'] = 'close_logic'

            if last_index_cond: df.at[i, 'action'] = 'last_index'
            if close_logic: df.at[i, 'action'] = 'close_logic'
            if tp_cond: df.at[i, 'action'] = 'profit_target'
            if sl_cond: df.at[i, 'action'] = 'stop_loss'

            df.at[i, 'close_price'] = now_close
            df.at[i, 'realized_pnl'] = realized_pnl
            df.at[i, 'commission'] = commission

        ##### record at last #####
        df.at[i, 'equity_value'] = equity_value
        df.at[i, 'num_of_share'] = num_of_coin
        df.at[i, 'unrealized_pnl'] = unrealized_pnl
        df.at[i, 'net_profit'] = net_profit

    if summary_mode:
        df = df[df['action'] != '']
    save_path = plotguy.generate_filepath(para_comb)
    print(save_path)
    df.to_parquet(save_path)


def get_hist_data(code_list, start_date):

    df_dict = {}
    for code in code_list:

        # ========== read csv file ==========
        project_root = Path(__file__).parent.parent.parent
        crypto_data_path = project_root / 'crypto_data' / f'{code}.csv'
        raw_df = pd.read_csv(crypto_data_path)
        raw_df['datetime'] = pd.to_datetime(raw_df['date'])
        raw_df = raw_df.set_index('datetime')

        df = raw_df.copy()
        df.index = df.index.strftime('%Y-%m-%d')
        df = df.loc[start_date:]

        df['pct'] = df['close'].pct_change()
        df_dict[code] = df

    return df_dict


def get_secondary_data(df_dict):

    for code, df in df_dict.items():
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['rsi-1'] = df['rsi'].shift(1)
        df['rsi_change'] = df['rsi'] - df['rsi-1']

        df_dict[code] = df

    return df_dict


def get_sec_profile(code_list, market, sectype, initial_capital):

    sec_profile = {}
    for code in code_list:
        symbol = code.split('_')[-1].split('.')[0]

    sec_profile['market'] = market
    sec_profile['sectype'] = sectype
    sec_profile['initial_capital'] = initial_capital
    sec_profile['code'] = code
    sec_profile['symbol'] = symbol
    if market == 'crypto' and sectype == 'perpetual':
        sec_profile['commission_rate'] = 0.0055 / 100

    return sec_profile


def get_all_para_comb(para_dict, df_dict, sec_profile, start_date, freq, output_folder, file_format,
                      summary_mode, py_filename):

    para_keys = list(para_dict.keys())
    para_values = list(para_dict.values())
    para_list = list(itertools.product(*para_values))

    intraday = True if freq != '1D' else False

    all_para_comb = []

    for reference_index in range(len(para_list)):
        para = para_list[reference_index]
        code = para[0]
        df = df_dict[code]

        para_comb = {}
        for i in range(len(para)):
            key = para_keys[i]
            para_comb[key] = para[i]

        para_comb['para_dict'] = para_dict
        para_comb['sec_profile'] = sec_profile
        para_comb['start_date'] = start_date
        para_comb['end_date'] = df.index[-1]
        para_comb['freq'] = freq
        para_comb['reference_index'] = reference_index
        para_comb['output_folder'] = output_folder
        para_comb['file_format'] = file_format
        para_comb['df'] = df
        para_comb['intraday'] = intraday
        para_comb['summary_mode'] = summary_mode
        para_comb['py_filename'] = py_filename

        all_para_comb.append(para_comb)

    return all_para_comb

if __name__ == '__main__':

    # ========== configuration ==========

    start_date = '2021-01-01'
    freq = '1D'
    market = 'crypto'
    sectype = 'perpetual'
    file_format = 'parquet'
    summary_mode = False
    num_of_core = 16
    mp_mode = True

    initial_capital = 10000

    parent_folder = '04_rsi'
    secondary_data_folder = os.path.join(parent_folder, '01_secondary_data')
    output_folder = os.path.join(parent_folder, '02_output_folder')
    file_format = 'parquet'
    py_filename = os.path.basename(__file__).replace('.py', '')

    if not os.path.exists(parent_folder): os.mkdir(parent_folder)
    if not os.path.exists(secondary_data_folder): os.mkdir(secondary_data_folder)
    if not os.path.exists(output_folder): os.mkdir(output_folder)

    code_list = [
        'GN01_market_price_usd_ohlc_24h_BTC',
        'GN02_market_price_usd_ohlc_24h_ETH'
    ]

    para_dict = {
        'code': code_list,
        'tp': [3, 5, 7, 10, 15, 20],
        'sl': [1, 2, 2.5, 5],
        'cycle': [10, 15, 20],

        'rsi_chg_dir': ['positive', 'negative'],
        'rsi_chg_thres': [3, 5, 10, 15],
    }


    df_dict = get_hist_data(code_list, start_date)

    df_dict = get_secondary_data(df_dict)

    sec_profile = get_sec_profile(code_list, market, sectype, initial_capital)

    all_para_comb = get_all_para_comb(para_dict, df_dict, sec_profile, start_date, freq, output_folder, file_format,
                                      summary_mode, py_filename)

    if mp_mode:
        pool = mp.Pool(processes=num_of_core)
        pool.map(backtest, all_para_comb)
        pool.close()
    else:
        for para_comb in all_para_comb:
            backtest(para_comb)

    plotguy.generate_backtest_result(
        all_para_combination=all_para_comb,
        number_of_core=num_of_core
    )

    app = plotguy.plot(
        mode='equity_curves',
        all_para_combination=all_para_comb
    )

    app.run_server(port=8900)