import pandas as pd
import numpy as np

# 1. 讀取你的 Backtest 報告
df = pd.read_csv('summary_20260311_001453.csv')

# 2. 將需要的參數設定為 Index，用來對齊 Long 和 Short
param_cols = ['code', 'std_ratio_thres', 'sma_len', 'sma_dir', 'candle_len', 'cycle', 'sl', 'tp']

# 將數據分為 Long (positive) 和 Short (negative)
df_long = df[df['candle_dir'] == 'positive'].set_index(param_cols)[['net_profit', 'num_of_trade']]
df_short = df[df['candle_dir'] == 'negative'].set_index(param_cols)[['net_profit', 'num_of_trade']]

# 3. 將 Long 和 Short 根據相同的參數合併 (Inner Join)
# 這樣可以確保這組參數必須同時產生 Long 和 Short 的信號
merged = df_long.join(df_short, lsuffix='_long', rsuffix='_short', how='inner')

# 4. 計算核心評估指標
# a. 總利潤
merged['total_profit'] = merged['net_profit_long'] + merged['net_profit_short']
# b. 總交易次數
merged['total_trades'] = merged['num_of_trade_long'] + merged['num_of_trade_short']
# c. Long / Short 比例 (最接近 1 為最佳)
# 為了方便篩選，我們計算: Min(Long, Short) / Max(Long, Short)
# 如果是 1:1，比例就是 1.0。如果是 2:1，比例就是 0.5。
merged['ls_ratio'] = np.minimum(merged['num_of_trade_long'], merged['num_of_trade_short']) / \
                     np.maximum(merged['num_of_trade_long'], merged['num_of_trade_short'])

# 5. 設定篩選條件 (The Quant Filter)
# 條件 A: 總利潤必須大於 0 (賺錢)
# 條件 B: Long:Short 比例大於 0.75 (即大約 1:1.3 以內，非常平衡)
# 條件 C: 總交易次數不能太少 (例如 4H 圖，2 年數據，最少要有 50 次交易才具備統計意義)
filtered = merged[
    (merged['total_profit'] > 0) &
    (merged['ls_ratio'] >= 0.75) &
    (merged['total_trades'] >= 50)
]

# 6. 找出最好的參數 (按總利潤排序)
best_params = filtered.sort_values(by='total_profit', ascending=False)

# 印出頭 10 個最完美平衡的參數組合！
print("=== 完美 Long:Short (1:1) 且總利潤最高的 10 組參數 ===")
print(best_params[['total_profit', 'net_profit_long', 'net_profit_short', 'total_trades', 'ls_ratio']].head(10))

# 輸出成 Excel 慢慢睇
best_params.to_csv('best_balanced_params.csv')