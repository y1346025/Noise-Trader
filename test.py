import gymnasium as gym
from stable_baselines3 import PPO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from market_env import MarketEnv

# --- 1. 設定 ---
model_path = "models/PPO-1767258295/final_model.zip" 
env = MarketEnv(sim_days=100) # [修正] 測試天數改為 100 天

# [重點] 模擬訓練後期，確保測試時有收手續費
env.total_steps_counter = env.curriculum_threshold + 1 
print(f"★ 測試模式啟動：模擬天數 100 天，已強制開啟 0.1% 手續費機制")

model = PPO.load(model_path, env=env)

# --- 2. 執行回測 ---
obs, info = env.reset()
done = False
history = {
    "day": [], "price": [], "assets": [], "action": [], "event": [],
    "pos_sent": [], "neg_sent": []
}

while not done:
    action, _ = model.predict(obs, deterministic=True)
    
    # 獲取環境內部的真實狀態
    real_env = env.unwrapped
    current_day = real_env.current_day
    current_assets = real_env.total_assets
    current_price = real_env._get_base_price()
    event_type, _ = real_env._get_current_event()
    
    # 提取情緒 (最後三維)
    obs_flat = obs.flatten()
    history["pos_sent"].append(obs_flat[7])
    history["neg_sent"].append(obs_flat[8])
    
    # 記錄數據
    history["day"].append(current_day)
    history["price"].append(current_price)
    history["assets"].append(current_assets)
    history["action"].append(["Buy", "Sell", "Hold"][action])
    history["event"].append(event_type)

    obs, reward, done, truncated, info = env.step(action)

# --- 3. 視覺化 (針對 100 天進行優化) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# 1. 資產曲線
ax1.plot(history["day"], history["assets"], color="#1f77b4", lw=2)
ax1.axhline(y=10000, color='red', ls='--', alpha=0.5, label="Initial Cash")
ax1.set_title("100-Day Backtest: Portfolio Value")
ax1.legend()

# 2. 價格與事件標記 (標記隨機出現的 Fake Panic)
ax2.plot(history["day"], history["price"], color="gray", alpha=0.5)
# 標記 Fake Panic
for d, e, p in zip(history["day"], history["event"], history["price"]):
    if e == "Fake_Panic":
        ax2.scatter(d, p, color="orange", marker="x", s=100)
    elif e == "Real_Bad":
        ax2.scatter(d, p, color="black", marker="o", s=50)

# 標記 AI 的買賣點
buy_days = [d for d, a in zip(history["day"], history["action"]) if a == "Buy"]
buy_px = [history["price"][history["day"].index(d)] for d in buy_days]
ax2.scatter(buy_days, buy_px, color="red", marker="^", label="Buy")

ax2.set_title("Stock Price & Events (X = Fake Panic, O = Real Bad)")
ax2.legend()

# 3. 情緒分佈
ax3.stackplot(history["day"], history["pos_sent"], history["neg_sent"], labels=['Positive', 'Negative'], colors=['green', 'red'], alpha=0.4)
ax3.set_title("NLP Sentiment Input Trend")
ax3.legend(loc='upper left')

plt.tight_layout()
plt.show()