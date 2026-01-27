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
timestamp_dir = "models/PPO-1769506790"  # 範例，請替換成你的實際路徑
model_path = f"{timestamp_dir}/final_model.zip"
stats_path = f"{timestamp_dir}/final_model_env.pkl"

if not os.path.exists(model_path) or not os.path.exists(stats_path):
    print(f"錯誤：找不到模型或環境參數檔。\n請確認路徑是否正確：{timestamp_dir}")
    exit()

print(f"--- 載入模型與正規化參數: {timestamp_dir} ---")

# --- 2. 重建環境與載入參數 ---
# 必須建立一個結構一模一樣的環境
env = DummyVecEnv([lambda: MarketEnv(sim_days=100, events_path='rich_events.json')])

# [關鍵] 載入訓練時的統計數據 (Mean/Std)
# 如果這一步沒做，AI 看到的 RSI=0.5 可能會被誤判成極大或極小值
env = VecNormalize.load(stats_path, env)

# 測試時不要更新統計數據 (Training=False)，也不用正規化 Reward
env.training = False 
env.norm_reward = False

# 強制開啟手續費 (透過訪問內層環境)
# env.envs[0] 是 DummyVecEnv 裡的第一個環境實體
env.envs[0].total_steps_counter = 30001 
print(f"★ 測試模式啟動：模擬天數 100 天，已強制開啟手續費機制")

model = PPO.load(model_path, env=env)

# --- 3. 執行回測 ---
obs = env.reset()
done = False
history = {
    "day": [], "price": [], "assets": [], "action": [], "event": [],
    "raw_sentiment": [] # 紀錄原始情緒 (Pos, Neg)
}

print("開始推論...")

# 因為 VecEnv 會自動 Reset，我們需要手動控制結束條件
while True:
    action, _ = model.predict(obs, deterministic=True)
    
    # --- 獲取真實數據用於繪圖 ---
    # 透過 env.envs[0] 取得最底層的 MarketEnv 實體
    real_env = env.envs[0]
    
    current_day = real_env.current_day
    current_assets = real_env.total_assets
    current_price = real_env._get_base_price()
    
    # 取得當前事件資料 (注意：MarketEnv 邏輯是 Bot 根據 current 反應，AI 預測 next)
    # 這裡我們記錄 "當下正在發生" 的事件
    current_event = real_env.current_event_data
    
    # 記錄數據
    history["day"].append(current_day)
    history["price"].append(current_price)
    history["assets"].append(current_assets)
    history["action"].append(["Buy", "Sell", "Hold"][action[0]]) # action 是 array
    history["event"].append(current_event["category"])
    history["raw_sentiment"].append(current_event["sentiment"]) # [Pos, Neg, Neu]

    # Step
    obs, reward, done, infos = env.step(action)
    
    # 檢查是否結束 (VecEnv 的 done 是 array)
    if done[0]:
        break

# --- 4. 視覺化 (針對隨機事件優化) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Graph 1: 資產曲線
ax1.plot(history["day"], history["assets"], color="#1f77b4", lw=2, label="AI Portfolio")
ax1.axhline(y=10000, color='red', ls='--', alpha=0.5, label="Initial Cash")
ax1.set_title("100-Day Backtest: Asset Growth (with 0.1% Fee)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Graph 2: 價格與關鍵事件
ax2.plot(history["day"], history["price"], color="gray", alpha=0.6, label="Price")

# 篩選並標記 "有發生事件" 的日子 (過濾掉 None)
event_days = []
event_prices = []
event_colors = []
event_markers = []

for d, e, p in zip(history["day"], history["event"], history["price"]):
    if e == "None": continue
    
    event_days.append(d)
    event_prices.append(p)
    
    if "Panic" in e or "Bad" in e:
        event_colors.append("red") # 壞消息
        event_markers.append("x")
    elif "Good" in e:
        event_colors.append("green") # 好消息
        event_markers.append("o")
    else:
        event_colors.append("orange")
        event_markers.append("s")

# 繪製事件點
for i in range(len(event_days)):
    ax2.scatter(event_days[i], event_prices[i], c=event_colors[i], marker=event_markers[i], s=80, zorder=5)

# 標記 AI 的買賣點
buy_days = [d for d, a in zip(history["day"], history["action"]) if a == "Buy"]
buy_px = [history["price"][history["day"].index(d)] for d in buy_days]
sell_days = [d for d, a in zip(history["day"], history["action"]) if a == "Sell"]
sell_px = [history["price"][history["day"].index(d)] for d in sell_days]

ax2.scatter(buy_days, buy_px, color="blue", marker="^", s=100, label="AI Buy")
ax2.scatter(sell_days, sell_px, color="black", marker="v", s=100, label="AI Sell")

ax2.set_title("Price Action, Events & AI Trades (Green=Good News, Red=Bad News)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Graph 3: 情緒流 (只畫 Pos/Neg，忽略 Neu)
# 轉換數據格式
pos_vals = [s[0] for s in history["raw_sentiment"]]
neg_vals = [s[1] for s in history["raw_sentiment"]]

ax3.bar(history["day"], pos_vals, color='green', alpha=0.5, label='Positive Sentiment')
ax3.bar(history["day"], [-v for v in neg_vals], color='red', alpha=0.5, label='Negative Sentiment')
ax3.axhline(0, color='black', lw=1)
ax3.set_title("Daily Market Sentiment (FinBERT)")
ax3.set_ylim(-1.0, 1.0)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()