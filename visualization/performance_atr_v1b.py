import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import re


# ==========================================
# 1. Filename Parser (ATR & Fixed % Compatible)
# ==========================================
def parse_filename(filename):
    """
    Parse filename and extract parameters
    Example: 03-1_pro_backtest_2026_02_24_BTC_4h_cdpos_cl1_sma10_wha_std1.0_tp7_sl2_cyc20_000161
    Example ATR: 03-4_pro_backtest_4h(atr)_2026_03_13_BTC_4h_cdneg_cl2_sma15_bel_std1_tpm3_slm2_cyc20_009791
    """
    # Remove .csv extension
    filename = filename.replace('.csv', '')
    parts = filename.split('_')

    # Find parameter positions
    symbol_idx = None
    resolution_idx = None

    for i, part in enumerate(parts):
        if part in ['BTC', 'ETH', 'SOL', 'SUI', 'DOGE']:  # Added common fleet symbols
            symbol_idx = i
            resolution_idx = i + 1
            break

    if symbol_idx is None:
        raise ValueError(f"Cannot identify symbol from filename: {filename}")

    # Extract base parameters
    params = {
        'file_prefix': '_'.join(parts[:symbol_idx + 1]),
        'date': f"{parts[2]}/{parts[3]}/{parts[4]}",
        'symbol': parts[symbol_idx],
        'resolution': parts[resolution_idx],
        'candle_dir': '',
        'candle_length': '',
        'sma': '',
        'position': '',
        'std': '',
        'take_profit': '',  # Unified key for both Fixed % and ATR
        'stop_loss': '',  # Unified key for both Fixed % and ATR
        'cycle': '',
        'ref_index': ''
    }

    # Parse remaining parameters dynamically
    remaining_parts = parts[resolution_idx + 1:]

    for part in remaining_parts:
        if part.startswith('cd'):
            params['candle_dir'] = part.replace('cd', '')
        elif part.startswith('cl'):
            params['candle_length'] = part.replace('cl', '') + '%'
        elif part.startswith('sma'):
            params['sma'] = part.replace('sma', '')
        elif part in ['wha', 'bel', 'abo']:
            params['position'] = part
        elif part.startswith('std'):
            params['std'] = part.replace('std', '')

        # 🛡️ 智能判斷 TP 類型 (ATR Multiplier vs Fixed %)
        elif part.startswith('tpm'):
            params['take_profit'] = part.replace('tpm', '') + 'x ATR'
        elif part.startswith('tp') and not part.startswith('tpm'):
            params['take_profit'] = part.replace('tp', '') + '%'

        # 🛡️ 智能判斷 SL 類型 (ATR Multiplier vs Fixed %)
        elif part.startswith('slm'):
            params['stop_loss'] = part.replace('slm', '') + 'x ATR'
        elif part.startswith('sl') and not part.startswith('slm'):
            params['stop_loss'] = part.replace('sl', '') + '%'

        elif part.startswith('cyc'):
            params['cycle'] = part.replace('cyc', '')
        elif part.isdigit() and len(part) >= 6:
            params['ref_index'] = part

    # Parse position description
    pos_map = {'wha': 'Whatever', 'bel': 'Below', 'abo': 'Above'}
    params['position_full'] = pos_map.get(params['position'], params['position'])

    return params


# ==========================================
# 2. Statistics Calculator
# ==========================================
def calculate_statistics(df):
    """Calculate key performance metrics"""
    # Initial capital
    initial_capital = 10000

    # Strategy returns
    strategy_returns = df['pct'].fillna(0)

    # Cumulative returns
    cum_returns = (1 + strategy_returns).cumprod()

    # Calculate final equity
    if 'equity_value' in df.columns:
        final_equity = df['equity_value'].iloc[-1]
    else:
        final_equity = initial_capital * cum_returns.iloc[-1]

    # Net Profit (in dollars and percentage)
    net_profit_dollars = final_equity - initial_capital
    net_profit_pct = (net_profit_dollars / initial_capital) * 100

    # Sharpe Ratio (annualized)
    if strategy_returns.std() != 0:
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(365 * 6)  # 4-hour candles
    else:
        sharpe = 0

    # Maximum Drawdown
    cummax = cum_returns.cummax()
    drawdown = (cum_returns / cummax - 1) * 100
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = abs(net_profit_pct / max_dd) if max_dd != 0 else 0

    # Count action types
    action_counts = {}
    if 'action' in df.columns:
        action_counts = {
            'open': (df['action'] == 'open').sum(),
            'profit_target': (df['action'] == 'profit_target').sum(),
            'stop_loss': (df['action'] == 'stop_loss').sum(),
            'close_logic': (df['action'] == 'close_logic').sum(),
            'total_trades': (df['action'] == 'open').sum()  # Number of trades = number of opens
        }
    else:
        action_counts = {
            'open': 0,
            'profit_target': 0,
            'stop_loss': 0,
            'close_logic': 0,
            'total_trades': 0
        }

    # Calculate Win Rate based on exit reasons
    total_completed_trades = action_counts['profit_target'] + action_counts['stop_loss'] + action_counts['close_logic']

    if total_completed_trades > 0:
        win_rate = (action_counts['profit_target'] / total_completed_trades) * 100
    else:
        win_rate = 0

    # Alternative win rate calculation if realized_pnl exists
    win_rate_by_pnl = 0
    if 'realized_pnl' in df.columns:
        profitable_trades = (df['realized_pnl'] > 0).sum()
        total_trades_with_pnl = (df['realized_pnl'] != 0).sum()
        win_rate_by_pnl = (profitable_trades / total_trades_with_pnl * 100) if total_trades_with_pnl > 0 else 0

    return {
        'net_profit_dollars': net_profit_dollars,
        'net_profit_pct': net_profit_pct,
        'sr': sharpe,
        'cr': calmar,
        'max_dd': max_dd,
        'action_counts': action_counts,
        'win_rate': win_rate,
        'win_rate_by_pnl': win_rate_by_pnl,
        'initial_capital': initial_capital,
        'final_equity': final_equity
    }


# ==========================================
# 3. Plotting Engine
# ==========================================
def plot_advanced_analysis(df, params, stats, output_filename):
    """
    Create professional-grade backtest analysis charts
    """
    # Set font to support special characters
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(36, 24))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 2, 2, 2], hspace=0.15)

    # Prepare data
    df['datetime_idx'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime_idx')

    # ========== Chart 1: Price & Trading Signals ==========
    ax1 = fig.add_subplot(gs[0])

    # Plot price
    ax1.plot(df.index, df['close'], color='darkgreen', linewidth=1.5, alpha=0.7, label='Price', zorder=1)

    # Mark entry points (open)
    if 'action' in df.columns:
        entry_points = df[df['action'] == 'open']
        if len(entry_points) > 0:
            ax1.scatter(entry_points.index, entry_points['close'],
                        marker='^', color='lime', s=150, label='Open',
                        edgecolors='darkgreen', linewidths=2, zorder=5)

        # Mark profit target exits
        profit_exits = df[df['action'] == 'profit_target']
        if len(profit_exits) > 0:
            ax1.scatter(profit_exits.index, profit_exits['close'],
                        marker='v', color='green', s=150, label='Profit Target',
                        edgecolors='darkgreen', linewidths=2, zorder=5)

        # Mark stop loss exits
        stop_exits = df[df['action'] == 'stop_loss']
        if len(stop_exits) > 0:
            ax1.scatter(stop_exits.index, stop_exits['close'],
                        marker='v', color='red', s=150, label='Stop Loss',
                        edgecolors='darkred', linewidths=2, zorder=5)

        # Mark close logic exits
        close_exits = df[df['action'] == 'close_logic']
        if len(close_exits) > 0:
            ax1.scatter(close_exits.index, close_exits['close'],
                        marker='v', color='orange', s=150, label='Close Logic',
                        edgecolors='darkorange', linewidths=2, zorder=5)

    # Title with all parameters (Auto-adapts to Fixed % or ATR)
    title_parts = [
        f"[{params['file_prefix']} | {params['symbol']} {params['resolution']}]",
        f"CD:{params['candle_dir']} CL:{params['candle_length']} SMA:{params['sma']} " +
        f"{params['position_full']} STD:{params['std']}",
        f"TP:{params['take_profit']} SL:{params['stop_loss']} CYC:{params['cycle']} REF:{params['ref_index']}"
    ]
    ax1.set_title('\n'.join(title_parts), fontsize=22, fontweight='bold', pad=25, family='monospace')

    ax1.set_ylabel('Price (USD)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')

    # ========== Chart 2: Close Price + SMA ==========
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot close price
    ax2.plot(df.index, df['close'], color='darkgreen', linewidth=1.5, alpha=0.7, label='Close Price', zorder=1)

    # Plot SMA if available
    if 'sma' in df.columns:
        ax2.plot(df.index, df['sma'], color='blue', linewidth=2,
                 label=f'SMA ({params["sma"]})', zorder=2)
        ax2.set_title(f"Close Price & Simple Moving Average (SMA {params['sma']})",
                      fontsize=18, fontweight='bold', pad=15)
    else:
        ax2.set_title("Close Price",
                      fontsize=18, fontweight='bold', pad=15)

    ax2.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')

    # ========== Chart 3: Equity Curve Comparison ==========
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Initial capital
    initial_capital = stats['initial_capital']

    # Strategy equity curve
    if 'equity_value' in df.columns:
        strategy_equity = df['equity_value']
    else:
        strategy_equity = initial_capital * (1 + df['pct'].fillna(0)).cumprod()

    # Buy & Hold equity curve
    first_price = df['close'].iloc[0]
    bh_equity = initial_capital * (df['close'] / first_price)

    # Calculate net profit
    strategy_net_profit = strategy_equity.iloc[-1] - initial_capital
    strategy_net_profit_pct = (strategy_net_profit / initial_capital) * 100

    bh_net_profit = bh_equity.iloc[-1] - initial_capital
    bh_net_profit_pct = (bh_net_profit / initial_capital) * 100

    # Plot equity curves
    ax3.plot(df.index, strategy_equity, color='black', linewidth=3,
             label=f'Strategy (Final: ${strategy_equity.iloc[-1]:.0f}, Net P/L: ${strategy_net_profit:+,.0f} [{strategy_net_profit_pct:+.1f}%])',
             zorder=3)
    ax3.plot(df.index, bh_equity, color='orange', linewidth=2.5,
             linestyle='--', alpha=0.8,
             label=f'Buy & Hold (Final: ${bh_equity.iloc[-1]:.0f}, Net P/L: ${bh_net_profit:+,.0f} [{bh_net_profit_pct:+.1f}%])',
             zorder=2)

    # Add horizontal line at initial capital
    ax3.axhline(y=initial_capital, color='gray', linestyle=':', linewidth=1.5,
                alpha=0.6, label=f'Initial Capital (${initial_capital:,})')

    ax3.set_title("Equity Curve Comparison", fontsize=18, fontweight='bold', pad=15)
    ax3.set_ylabel('Portfolio Value (USD)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor('#f8f9fa')

    # Format Y-axis to show currency
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Statistics info box
    action_counts = stats['action_counts']
    stats_text = (f"Sharpe: {stats['sr']:.2f} | Calmar: {stats['cr']:.2f} | "
                  f"Trades: {action_counts['total_trades']} | Win Rate: {stats['win_rate']:.1f}%\n"
                  f"TP: {action_counts['profit_target']} | SL: {action_counts['stop_loss']} | "
                  f"CL: {action_counts['close_logic']}")
    ax3.text(0.98, 0.05, stats_text, transform=ax3.transAxes,
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', boxstyle='round,pad=0.8'),
             fontsize=13, ha='right', va='bottom', fontweight='bold', family='monospace')

    # ========== Chart 4: Drawdown Analysis ==========
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # Strategy drawdown
    strat_cummax = strategy_equity.cummax()
    strat_dd = (strategy_equity / strat_cummax - 1) * 100

    # Buy & Hold drawdown
    bh_cummax = bh_equity.cummax()
    bh_dd = (bh_equity / bh_cummax - 1) * 100

    # Plot with proper visibility
    ax4.fill_between(df.index, strat_dd, 0, color='red', alpha=0.4, label='Strategy DD', zorder=2)
    ax4.plot(df.index, bh_dd, color='orange', linestyle='--', linewidth=2.5,
             alpha=0.9, label='B&H DD', zorder=3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    ax4.set_title("Drawdown Analysis (%)", fontsize=18, fontweight='bold', pad=15)
    ax4.set_ylabel('Drawdown %', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower left', ncol=2, fontsize=11)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor('#f8f9fa')

    # Drawdown statistics annotations
    ax4.text(0.99, 0.15, f"Strategy MaxDD: {strat_dd.min():.2f}%",
             transform=ax4.transAxes, color='darkred', fontweight='bold',
             fontsize=12, ha='right',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    ax4.text(0.99, 0.85, f"B&H MaxDD: {bh_dd.min():.2f}%",
             transform=ax4.transAxes, color='darkorange', fontweight='bold',
             fontsize=12, ha='right',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Format X-axis dates
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Save chart
    plt.savefig(output_filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Chart saved: {output_filename}")


# ==========================================
# 4. Main Program
# ==========================================
def main(csv_file):
    """Main execution function"""
    # Read data
    print(f"📂 Reading file: {csv_file}")
    df = pd.read_csv(csv_file)

    # Parse filename
    filename = Path(csv_file).stem
    params = parse_filename(filename)

    # Calculate statistics
    print("📊 Calculating performance metrics...")
    stats = calculate_statistics(df)

    # Output filename
    output_file = f"result_chris/{filename}_analysis.png"

    # Generate charts
    print("🎨 Generating visualization charts...")
    plot_advanced_analysis(df, params, stats, output_file)

    # Print statistics summary
    action_counts = stats['action_counts']
    print("\n" + "=" * 60)
    print("📊 Backtest Performance Summary")
    print("=" * 60)
    print(f"Strategy Parameters: {params['symbol']} {params['resolution']}")
    print(f"Candle Direction: {params['candle_dir']} | Length: {params['candle_length']}")
    print(f"SMA: {params['sma']} | Position: {params['position_full']} | STD: {params['std']}")
    print(f"Target (TP): {params['take_profit']} | Risk (SL): {params['stop_loss']} | Cycle: {params['cycle']}")
    print("-" * 60)
    print("📈 Performance Metrics:")
    print(f"Initial Capital: ${stats['initial_capital']:,.2f}")
    print(f"Final Equity: ${stats['final_equity']:,.2f}")
    print(f"Net Profit: ${stats['net_profit_dollars']:+,.2f} ({stats['net_profit_pct']:+.2f}%)")
    print(f"Sharpe Ratio: {stats['sr']:.2f}")
    print(f"Calmar Ratio: {stats['cr']:.2f}")
    print(f"Maximum Drawdown: {stats['max_dd']:.2f}%")
    print("-" * 60)
    print("🔄 Trading Activity:")
    print(f"Total Trades (Opens): {action_counts['total_trades']}")
    print(f"  ├─ Profit Targets: {action_counts['profit_target']}")
    print(f"  ├─ Stop Losses: {action_counts['stop_loss']}")
    print(f"  └─ Close Logic: {action_counts['close_logic']}")
    print(f"Win Rate (TP/Total): {stats['win_rate']:.1f}%")
    if stats['win_rate_by_pnl'] > 0:
        print(f"Win Rate (by P&L): {stats['win_rate_by_pnl']:.1f}%")
    print("=" * 60 + "\n")


# ==========================================
# 5. Execution Example
# ==========================================
if __name__ == "__main__":
    # Use your CSV file path
    csv_file = "result_chris/03-4_pro_backtest_4h(atr)_2026_03_13_BTC_4h_cdneg_cl2_sma15_bel_std1_tpm3_slm2_cyc20_009791.csv"
    main(csv_file)