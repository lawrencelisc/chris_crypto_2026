import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np
import multiprocessing as mp

import plotguy
import itertools

from pathlib import Path


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def generate_filepath(para_comb):
    """
    Generate filepath for saving backtest results
    """
    output_folder = para_comb['output_folder']
    file_format = para_comb['file_format']
    py_filename = para_comb['py_filename']
    reference_index = para_comb['reference_index']

    # Extract key parameters for filename
    code = para_comb['code']
    candle_dir = para_comb['candle_dir']
    candle_len = para_comb['candle_len']
    sma_len = para_comb['sma_len']
    sma_dir = para_comb['sma_dir']
    std_ratio_thres = para_comb['std_ratio_thres']
    tp = para_comb['tp']
    sl = para_comb['sl']
    cycle = para_comb['cycle']
    freq = para_comb['freq']

    # Extract symbol from code (e.g., 'BTC' from 'GN01_market_price_usd_ohlc_4h_BTC')
    symbol = code.split('_')[-1]

    # Create filename with parameters
    filename = (
        f"{py_filename}_{symbol}_{freq}_"
        f"cd{candle_dir[:3]}_cl{candle_len}_"
        f"sma{sma_len}_{sma_dir[:3]}_"
        f"std{std_ratio_thres}_"
        f"tp{tp}_sl{sl}_cyc{cycle}_"
        f"{reference_index:06d}.{file_format}"
    )

    # Create full path
    save_path = os.path.join(output_folder, filename)

    return save_path


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
    candle_dir      = para_comb['candle_dir']
    candle_len      = para_comb['candle_len']
    sma_len         = para_comb['sma_len']
    sma_dir         = para_comb['sma_dir']
    std_ratio_thres = para_comb['std_ratio_thres']

    tp              = para_comb['tp']
    sl              = para_comb['sl']

    cycle           = para_comb['cycle']

    # ========== secondary profile ==========
    market          = sec_profile['market']
    sectype         = sec_profile['sectype']
    initial_capital = sec_profile['initial_capital']
    commission_rate = sec_profile['commission_rate']
    slippage_rate   = sec_profile['slippage_rate']

    # ========== strategy specific ==========

    df = df.copy()
    df['sma'] = df['close'].rolling(sma_len).mean()
    df['std'] = df['close'].rolling(sma_len).std()
    df['std_raito'] = (df['sma'] - df['close']) / df['std']

    # ========== initialization (function) ==========

    df['action'] = ''
    df['num_of_share'] = 0

    df['open_price'] = np.NaN
    df['close_price'] = np.NaN

    df['realized_pnl'] = np.NaN
    df['dir'] = 0
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

    current_dir = 0
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

        now_candle = row['candle']
        now_sma = row['sma']
        now_std_raito = row['std_raito']

        ##### commission #####
        if num_of_coin > 0:
            open_fee = (num_of_coin * open_price) * commission_rate
            close_fee = (num_of_coin * now_close) * commission_rate
            commission = open_fee + close_fee

            open_slippage = (num_of_coin * open_price) * slippage_rate
            close_slippage = (num_of_coin * now_close) * slippage_rate
            slippage_cost = open_slippage + close_slippage

            # 3. 總摩擦成本
            total_friction = commission + slippage_cost
        else:
            total_friction = 0

        ##### equity value #####
        unrealized_pnl = num_of_coin * (now_close - open_price) * current_dir - total_friction
        equity_value   = last_realized_capital + unrealized_pnl
        net_profit     = round(equity_value - initial_capital, 2)

        trade_logic = False
        signal_dir = 0

        if candle_dir == 'positive':
            trade_logic = now_candle > candle_len * 0.01
            signal_dir = 1
        elif candle_dir == 'negative':
            trade_logic = now_candle < -1 * candle_len * 0.01
            signal_dir = -1

        if sma_dir == 'above':
            trade_logic = trade_logic and (now_close > now_sma) and now_std_raito < -1 * std_ratio_thres
        elif sma_dir == 'below':
            trade_logic = trade_logic and (now_close < now_sma) and now_std_raito > std_ratio_thres

        if trade_logic: df.at[i, 'logic'] = 'trade_logic'

        t_diff_hr = (now_t - open_t).total_seconds() / 3600
        if (freq == '1D'):
            num_of_res = 24
        else:
            num_of_res = int(freq[:-1])  # num of hours per cycle
        close_logic = (t_diff_hr / num_of_res) >= cycle
        tp_cond = open_price != 0 and ((now_close - open_price) * current_dir) > (tp * 0.01 * open_price)
        sl_cond = open_price != 0 and ((open_price - now_close) * current_dir) > (sl * 0.01 * open_price)
        last_index_cond = i == df.index[-1]

        ##### open position #####
        if num_of_coin == 0 and not last_index_cond and trade_logic:

            num_of_coin = last_realized_capital / now_close

            open_price = now_close
            open_t = now_t
            current_dir = signal_dir

            df.at[i, 'action'] = 'open'
            df.at[i, 'open_price'] = open_price

        ##### close position #####
        elif num_of_coin > 0 and (tp_cond or sl_cond or last_index_cond or close_logic):

            realized_pnl = unrealized_pnl
            unrealized_pnl = 0
            last_realized_capital += realized_pnl

            num_of_trade += 1
            num_of_coin = 0
            current_dir = 0

            if close_logic: df.at[i, 'logic'] = 'close_logic'

            if last_index_cond: df.at[i, 'action'] = 'last_index'
            if close_logic: df.at[i, 'action'] = 'close_logic'
            if tp_cond: df.at[i, 'action'] = 'profit_target'
            if sl_cond: df.at[i, 'action'] = 'stop_loss'

            df.at[i, 'close_price'] = now_close
            df.at[i, 'realized_pnl'] = realized_pnl
            df.at[i, 'commission'] = total_friction

        ##### record at last #####
        df.at[i, 'equity_value'] = equity_value
        df.at[i, 'num_of_share'] = num_of_coin
        df.at[i, 'unrealized_pnl'] = unrealized_pnl
        df.at[i, 'net_profit'] = net_profit
        df.at[i, 'dir'] = current_dir

    # if summary_mode:
    #     df = df[df['action'] != '']

    save_path = generate_filepath(para_comb)
    print(f'Saving to: {save_path}')
    df.reset_index(inplace=True)
    df.to_csv(save_path)

    return {
        'reference_index': para_comb['reference_index'],
        'net_profit': net_profit,
        'num_of_trade': num_of_trade,
        'code': code,
        'std_ratio_thres': std_ratio_thres,
        'sma_len': sma_len,
        'sma_dir': sma_dir,
        'candle_dir': candle_dir,
        'candle_len': candle_len,
        'cycle': cycle,
        'sl': sl,
        'tp': tp
    }


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
        df = df.loc[start_date:]

        df['pct'] = df['close'].pct_change()

        df_dict[code] = df

    return df_dict


def get_secondary_data(df_dict):

    for code, df in df_dict.items():
        df['candle'] = df['close'] / df['open'] - 1
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
        sec_profile['commission_rate'] = 0.055 / 100
        sec_profile['slippage_rate'] = 0.05 / 100

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

    start_date = '2024-01-01'
    freq = '4h'
    market = 'crypto'
    sectype = 'perpetual'
    summary_mode = False
    num_of_core = 16
    mp_mode = True

    initial_capital = 10000

    file_format = 'csv'
    secondary_data_folder = '01_secondary_data'
    output_folder = '02_output_folder'

    py_filename = os.path.basename(__file__).replace('.py', '')

    if not os.path.exists(secondary_data_folder): os.mkdir(secondary_data_folder)
    if not os.path.exists(output_folder): os.mkdir(output_folder)

    code_list = [
        'GN01_market_price_usd_ohlc_4h_BTC',
        'GN02_market_price_usd_ohlc_4h_ETH',
        'GN03_market_price_usd_ohlc_4h_SOL',
        'GN04_market_price_usd_ohlc_4h_SUI',
        'GN05_market_price_usd_ohlc_4h_DOGE'
    ]

    para_dict = {
        'code': code_list,
        'candle_dir': ['positive', 'negative'],
        'candle_len': [1, 2, 4],
        'sma_len': [10, 15, 20],
        'sma_dir': ['above', 'below', 'whatever'],
        'std_ratio_thres': [1.0, 2.0, 2.5],
        'tp': [3, 5, 7, 10],
        'sl': [1, 2],
        'cycle': [10, 15, 20],
    }


    df_dict = get_hist_data(code_list, start_date)

    df_dict = get_secondary_data(df_dict)

    sec_profile = get_sec_profile(code_list, market, sectype, initial_capital)

    all_para_comb = get_all_para_comb(para_dict, df_dict, sec_profile, start_date, freq, output_folder, file_format,
                                      summary_mode, py_filename)

    if mp_mode:
        pool = mp.Pool(processes=num_of_core)
        all_results = pool.map(backtest, all_para_comb)
        pool.close()
        pool.join()
    else:
        for para_comb in all_para_comb:
            result = backtest(para_comb)
            all_results.append(result)

    # 整理成 DataFrame
    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values(by='net_profit', ascending=False)

    # 打印摘要
    print("\n" + "=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"\nTotal combinations tested: {len(result_df)}")
    print(f"Best net profit: ${result_df['net_profit'].max():,.2f}")
    print(f"Worst net profit: ${result_df['net_profit'].min():,.2f}")
    print(f"Average net profit: ${result_df['net_profit'].mean():,.2f}")
    print(f"\nTop 10 results:")
    print(result_df.head(10).to_string())

    # 保存 CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(output_folder, f"summary_{timestamp}.csv")
    result_df.to_csv(summary_file, index=False)
    print(f"\nResults saved to: {summary_file}")
    print("=" * 50)