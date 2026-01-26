import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor # [新增] 用於監控
import os
import time

# 引入您剛寫好的環境 (確保 market_env.py 是最新版包含 RSI/MACD 的代碼)
from market_env import MarketEnv

# --- 設定儲存路徑 ---
models_dir = f"models/PPO-{int(time.time())}"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def main():
    print("--- 1. 初始化環境 ---")
    # 建立環境
    env = MarketEnv(sim_days=100)
    
    # [新增] 包裝 Monitor，這樣 TensorBoard 才能正確記錄每一局的 Reward
    env = Monitor(env, log_dir)

    print("--- 2. 建立 AI 模型 (Agent) ---")
    # 參數建議：
    # n_steps: 設為 2048 或更高，因為 MACD 需要看長期的歷史，太短的採樣會切斷關聯性
    # learning_rate: 稍微調低一點點，讓它學得穩一點
    model = PPO(
        "MlpPolicy", 
        env, 
        device="auto",
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003, 
        n_steps=2048,         
        batch_size=64,
        ent_coef=0.01,        # [建議] 稍微提高熵係數，防止它太早決定"躺平"
        gamma=0.95            # [建議] 提高 gamma，讓它更重視未來的長期獲利 (不僅是當下)
    )

    print("--- 3. 開始訓練 (Training) ---")
    
    # 課程學習策略：
    # Environment 裡設定了前 20,000 步免手續費。
    # 所以我們至少要訓練超過這個數字，它才會經歷「天堂 -> 地獄」的轉變。
    
    TOTAL_TIMESTEPS = 200000 #  20~30 萬步 看起來不錯
    SAVE_INTERVAL = 50000    # 每 5 萬步存一次
    
    current_step = 0
    while current_step < TOTAL_TIMESTEPS:
        model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False, tb_log_name="PPO_Trader")
        current_step += SAVE_INTERVAL
        
        model.save(f"{models_dir}/{current_step}")
        print(f"--- 目前進度 {current_step} 步，模型已儲存 ---")

    print("--- 訓練結束 ---")
    model.save(f"{models_dir}/final_model")
    print(f"最終模型已儲存至: {models_dir}/final_model.zip")

if __name__ == "__main__":
    main()