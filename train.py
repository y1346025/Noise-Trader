import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # [新增] 正規化工具
import os
import time
import torch

# 引入環境
from market_env import MarketEnv

# --- 設定儲存路徑 ---
timestamp = int(time.time())
models_dir = f"models/PPO-{timestamp}"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def main():
    print(f"--- 1. 初始化環境 (macOS 15 / MPS check) ---")
    
    # 檢查是否能用 MPS (雖然 PPO 策略網路較小，跑 CPU 可能更快，但確認一下無妨)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"PPO Training Device: {device}")

    # 建立環境：確保讀取新的 rich_events.json
    # 注意：這裡用 Lambda 函數包裝，是為了配合 DummyVecEnv
    env_maker = lambda: Monitor(MarketEnv(sim_days=100, events_path='rich_events.json'), log_dir)
    
    # [關鍵修正] 向量化環境 + 正規化
    # 1. DummyVecEnv: 讓 SB3 可以處理環境
    # 2. VecNormalize: 自動將 Observation 和 Reward 縮放到標準常態分佈，這對金融數據極度重要！
    env = DummyVecEnv([env_maker])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    print("--- 2. 建立 AI 模型 (Agent) ---")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        device="auto", # 讓 SB3 自動決定 (通常 Mac 上小模型跑 CPU 效率反而好，因為省去數據搬運)
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003, 
        n_steps=2048,         
        batch_size=64,
        ent_coef=0.01,        # 稍微降低一點，因為隨機事件已經提供了足夠的環境變異
        gamma=0.99,           # [重要] 提高到 0.99，讓 AI 願意持有股票等待 25% 的事件發生
        gae_lambda=0.95,
        clip_range=0.2,
    )

    print("--- 3. 開始訓練 (Training) ---")
    
    # 總步數設定
    # 因為隨機性增加，建議跑多一點步數讓它收斂
    TOTAL_TIMESTEPS = 500000 
    SAVE_INTERVAL = 50000    
    
    current_step = 0
    while current_step < TOTAL_TIMESTEPS:
        model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False, tb_log_name="PPO_RichEvent")
        current_step += SAVE_INTERVAL
        
        # 儲存模型
        model_path = f"{models_dir}/{current_step}.zip"
        model.save(model_path)
        
        # [關鍵] 同時儲存環境的正規化統計數據 (Running Mean/Std)
        # 如果沒有存這個，讀取模型時 AI 會變成「瞎子」，因為它看到的數值範圍跟訓練時不一樣
        env.save(f"{models_dir}/{current_step}_env.pkl")
        
        print(f"--- 進度: {current_step} 步 | 模型與正規化參數已儲存 ---")

    print("--- 訓練結束 ---")
    model.save(f"{models_dir}/final_model")
    env.save(f"{models_dir}/final_model_env.pkl")
    print(f"訓練好模型已儲存至: {models_dir}")

if __name__ == "__main__":
    main()