import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 引入你的環境
from market_env import MarketEnv

# --- 1. 設定路徑與參數 ---
# [請修改] 這裡要換成你剛剛訓練出來的資料夾名稱
timestamp_dir = "models/PPO-1769597396" 
model_path = f"{timestamp_dir}/final_model.zip"
stats_path = f"{timestamp_dir}/final_model_env.pkl"

if not os.path.exists(model_path) or not os.path.exists(stats_path):
    print(f"錯誤：找不到模型或環境參數檔。\n請確認路徑是否正確：{timestamp_dir}")
    exit()

print(f"--- 載入模型與正規化參數: {timestamp_dir} ---")

# --- 2. 重建環境與載入參數 ---
# 必須建立一個結構一模一樣的環境
# [注意] 這裡將 sim_days 改回 100 以配合圖表標題，若需 200 請自行修改
env = DummyVecEnv([lambda: MarketEnv(sim_days=200, events_path='rich_events.json')])

# [關鍵] 載入訓練時的統計數據 (Mean/Std)
env = VecNormalize.load(stats_path, env)

# 測試時不要更新統計數據 (Training=False)，也不用正規化 Reward
env.training = False 
env.norm_reward = False

# 強制開啟手續費 (透過訪問內層環境)
env.envs[0].total_steps_counter = 30001 
print(f"★ 測試模式啟動：模擬天數 100 天，已強制開啟手續費機制")

model = PPO.load(model_path, env=env)

# --- 3. 執行回測 ---
obs = env.reset()
done = False
history = {
    "day": [], "price": [], "assets": [], "action": [], "event": [],
    "raw_sentiment": [] 
}

print("開始推論...")

while True:
    action, _ = model.predict(obs, deterministic=True)
    
    real_env = env.envs[0]
    
    current_day = real_env.current_day
    current_assets = real_env.total_assets
    current_price = real_env._get_base_price()
    current_event = real_env.current_event_data
    
    history["day"].append(current_day)
    history["price"].append(current_price)
    history["assets"].append(current_assets)
    history["action"].append(["Buy", "Sell", "Hold"][action[0]]) 
    history["event"].append(current_event["category"])
    history["raw_sentiment"].append(current_event["sentiment"]) 

    obs, reward, done, infos = env.step(action)
    
    if done[0]:
        break

# --- 4. 視覺化 (高度優化版) ---
from matplotlib.lines import Line2D # [新增] 用於自定義圖例

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)

# === Graph 1: 資產曲線 ===
ax1.plot(history["day"], history["assets"], color="#1f77b4", lw=2, label="AI Portfolio")
ax1.axhline(y=10000, color='red', ls='--', alpha=0.5, label="Initial Cash")
ax1.set_title("200-Day Backtest: Asset Growth (0.1% Fee Included)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left")

# === Graph 2: 價格、交易與詳細事件 ===
# 2.1 畫股價
ax2.plot(history["day"], history["price"], color="silver", lw=1.5, label="Stock Price", zorder=1)

# 2.2 準備事件資料容器
event_handles = {} # 用來存圖例的樣式

for d, e, p in zip(history["day"], history["event"], history["price"]):
    if e == "None": continue
    
    # --- 事件樣式定義邏輯 ---
    # 形狀: o=Good, x=Bad/Panic, s=Neutral
    # 顏色: Green/Red=Real, Orange=Fake/Rumor, Gray=Neutral
    
    marker = 's'
    color = 'gray'
    label = e
    size = 80
    
    if "Real_Good" in e:
        marker, color, label = 'o', 'blue', 'Real Good (Official)'
    elif "Fake_Good" in e:
        marker, color, label = 'o', 'mediumpurple', 'Fake Good (Hype)'
    elif "Real_Bad" in e:
        marker, color, label = 'x', 'red', 'Real Bad (Crash)'
    elif "Fake_Panic" in e:
        marker, color, label = 'x', 'orange', 'Fake Panic (Rumor)'
    elif "Neutral" in e:
        marker, color, label = 's', 'gray', 'Neutral / Noise'

    # 繪製事件點
    ax2.scatter(d, p, c=color, marker=marker, s=size, zorder=3, alpha=0.9)
    
    # 收集圖例 (避免重複)
    if label not in event_handles:
        event_handles[label] = Line2D([0], [0], color='w', markerfacecolor=color, 
                                      marker=marker, markeredgecolor=color, 
                                      markersize=10, label=label)

# 2.3 繪製 AI 買賣點 (配色修改：Buy=Blue, Sell=Black)
buy_days = [d for d, a in zip(history["day"], history["action"]) if a == "Buy"]
buy_px = [history["price"][history["day"].index(d)] for d in buy_days]
sell_days = [d for d, a in zip(history["day"], history["action"]) if a == "Sell"]
sell_px = [history["price"][history["day"].index(d)] for d in sell_days]

ax2.scatter(buy_days, buy_px, color="red", marker="^", s=120, zorder=5, label="Buy")
ax2.scatter(sell_days, sell_px, color="forestgreen", marker="v", s=120, zorder=5, label="Sell")

# 2.4 製作精美的自定義圖例
# 合併交易圖例與事件圖例
custom_lines = [
    Line2D([0], [0], color='w', marker='^', markerfacecolor='red', markersize=10, label='Buy'),
    Line2D([0], [0], color='w', marker='v', markerfacecolor='forestgreen', markersize=10, label='Sell'),
] + list(event_handles.values())

ax2.set_title("Market Events & Trading Actions")
ax2.legend(handles=custom_lines, loc='upper left', ncol=2, fontsize=10)
ax2.grid(True, alpha=0.3)

# === Graph 3: 情緒流 (FinBERT) ===
pos_vals = [s[0] for s in history["raw_sentiment"]]
neg_vals = [s[1] for s in history["raw_sentiment"]]

ax3.bar(history["day"], pos_vals, color='forestgreen', alpha=0.6, label='Positive Sentiment')
ax3.bar(history["day"], [-v for v in neg_vals], color='firebrick', alpha=0.6, label='Negative Sentiment')
ax3.axhline(0, color='black', lw=0.8)
ax3.set_title("Daily Sentiment Analysis (FinBERT Output)")
ax3.set_ylabel("Sentiment Score")
ax3.set_ylim(-1.0, 1.0)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()