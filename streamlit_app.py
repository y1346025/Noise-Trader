import streamlit as st
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob
import datetime

from market_env import MarketEnv

# --- 初始化歷史紀錄 ---
if 'history_list' not in st.session_state:
    st.session_state.history_list = []

# --- 側邊欄：清除歷史紀錄按鈕 ---
if st.sidebar.button("清除所有歷史紀錄"):
    st.session_state.history_list = []
    st.rerun()

# --- 0. 自動偵測模型邏輯 ---
def get_model_folders():
    base_dir = "models"
    if not os.path.exists(base_dir):
        return []
    folders = [os.path.basename(f) for f in glob.glob(os.path.join(base_dir, "PPO-*"))]
    # 按照資料夾建立時間排序，最新的在前
    folders.sort(key=lambda x: os.path.getctime(os.path.join(base_dir, x)), reverse=True)
    return folders

# --- 1. 核心功能函數 (必須放在按鈕邏輯之前，解決 NameError) ---
@st.cache_resource
def load_model_and_env(model_dir, sim_days):
    timestamp_dir = f"models/{model_dir}"
    model_path = f"{timestamp_dir}/final_model.zip"
    stats_path = f"{timestamp_dir}/final_model_env.pkl"
    
    # 檢查檔案是否存在，避免 FileNotFoundError
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        return None, None, f"找不到檔案：{model_dir}"
    
    # 建立環境
    temp_env = DummyVecEnv([lambda: MarketEnv(sim_days=sim_days, events_path='rich_events.json')])
    temp_env = VecNormalize.load(stats_path, temp_env)
    temp_env.training = False 
    temp_env.norm_reward = False
    
    model = PPO.load(model_path, env=temp_env)
    return model, temp_env, None

# --- 2. 頁面配置 ---
st.set_page_config(page_title="NoiseTrader Dashboard", layout="wide")
st.title("📊 NoiseTrader: AI 交易決策儀表板")

# --- 3. 側邊欄控制 ---
st.sidebar.header("環境參數設定")
model_list = get_model_folders()

if not model_list:
    st.sidebar.error("❌ 找不到 models/ 資料夾或 PPO 模型")
    selected_model_dir = None
else:
    selected_model_dir = st.sidebar.selectbox("選擇訓練模型版本", model_list, index=0)

sim_days = st.sidebar.slider("回測模擬天數", 50, 500, 200)
fee_enabled = st.sidebar.checkbox("啟用交易手續費 (0.1%)", value=True)

# --- 4. 執行回測邏輯 ---
if st.sidebar.button("開始執行回測") and selected_model_dir:
    with st.spinner('AI 正在讀取模型並計算策略...'):
        model, env, error_msg = load_model_and_env(selected_model_dir, sim_days)
        
        if error_msg:
            st.error(error_msg)
        else:
            # 強制手續費機制 (透過訪問內層環境)
            if fee_enabled:
                env.envs[0].total_steps_counter = 30001 

            obs = env.reset()
            history = {"day": [], "price": [], "assets": [], "action": [], "event": [], "raw_sentiment": []}

            # 開始推論迴圈
            while True:
                action, _ = model.predict(obs, deterministic=True)
                real_env = env.envs[0]
                
                history["day"].append(real_env.current_day)
                history["price"].append(real_env._get_base_price())
                history["assets"].append(real_env.total_assets)
                history["action"].append(["Buy", "Sell", "Hold"][action[0]]) 
                history["event"].append(real_env.current_event_data["category"])
                history["raw_sentiment"].append(real_env.current_event_data["sentiment"]) 

                obs, reward, done, infos = env.step(action)
                if done[0]: break

            # --- A. 顯示 KPI ---
            final_assets = history["assets"][-1]
            total_return = (final_assets - 10000) / 10000 * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("最終資產 (USD)", f"${final_assets:,.2f}")
            c2.metric("總報酬率 (%)", f"{total_return:.2f}%", delta=f"{total_return:.2f}%")
            c3.metric("測試模型時間戳", selected_model_dir.split('-')[-1])

            # --- B. 視覺化繪圖 (移植 test.py 的優化邏輯) ---
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)

            # Graph 1: 資產成長
            ax1.plot(history["day"], history["assets"], color="#1f77b4", lw=2, label="AI Portfolio")
            ax1.axhline(y=10000, color='red', ls='--', alpha=0.5, label="Initial Cash")
            ax1.set_title(f"Backtest: {sim_days}-Day Asset Growth")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper left")

            # Graph 2: 價格、交易與詳細事件 (最重要的一環)
            ax2.plot(history["day"], history["price"], color="silver", lw=1.5, label="Stock Price", zorder=1)
            
            event_handles = {} 
            for d, e, p in zip(history["day"], history["event"], history["price"]):
                if e == "None": continue
                marker, color, label, size = 's', 'gray', e, 80
                if "Real_Good" in e: marker, color, label = 'o', 'blue', 'Real Good (Official)'
                elif "Fake_Good" in e: marker, color, label = 'o', 'mediumpurple', 'Fake Good (Hype)'
                elif "Real_Bad" in e: marker, color, label = 'x', 'red', 'Real Bad (Crash)'
                elif "Fake_Panic" in e: marker, color, label = 'x', 'orange', 'Fake Panic (Rumor)'
                elif "Neutral" in e: marker, color, label = 's', 'gray', 'Neutral / Noise'

                ax2.scatter(d, p, c=color, marker=marker, s=size, zorder=3, alpha=0.8)
                if label not in event_handles:
                    event_handles[label] = Line2D([0], [0], color='w', markerfacecolor=color, 
                                                  marker=marker, markeredgecolor=color, markersize=10, label=label)

            # 買賣點標註
            buy_days = [d for d, a in zip(history["day"], history["action"]) if a == "Buy"]
            buy_px = [history["price"][history["day"].index(d)] for d in buy_days]
            sell_days = [d for d, a in zip(history["day"], history["action"]) if a == "Sell"]
            sell_px = [history["price"][history["day"].index(d)] for d in sell_days]

            ax2.scatter(buy_days, buy_px, color="red", marker="^", s=120, zorder=5, label="Buy")
            ax2.scatter(sell_days, sell_px, color="forestgreen", marker="v", s=120, zorder=5, label="Sell")

            # 合併圖例
            custom_lines = [
                Line2D([0], [0], color='w', marker='^', markerfacecolor='red', markersize=10, label='Buy'),
                Line2D([0], [0], color='w', marker='v', markerfacecolor='forestgreen', markersize=10, label='Sell'),
            ] + list(event_handles.values())
            ax2.legend(handles=custom_lines, loc='upper left', ncol=2)
            ax2.set_title("Market Events & AI Trading Decisions")
            ax2.grid(True, alpha=0.3)

            # Graph 3: 情緒流
            pos_vals = [s[0] for s in history["raw_sentiment"]]
            neg_vals = [s[1] for s in history["raw_sentiment"]]
            ax3.bar(history["day"], pos_vals, color='forestgreen', alpha=0.6, label='Pos')
            ax3.bar(history["day"], [-v for v in neg_vals], color='firebrick', alpha=0.6, label='Neg')
            ax3.axhline(0, color='black', lw=0.8)
            ax3.set_ylim(-1.0, 1.0)
            ax3.set_title("Daily Sentiment (FinBERT)")
            ax3.legend(loc='upper right')

            plt.tight_layout()
            st.pyplot(fig) # 在 Streamlit 顯示圖表

            # --- C. 原始數據表格 ---
            with st.expander("🔍 點擊展開：查看詳細交易日誌"):
                st.dataframe(pd.DataFrame(history), use_container_width=True)
            
            # --- 存檔到 Session State ---
            record = {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "model": selected_model_dir,
                "days": sim_days,
                "fee": "Enabled" if fee_enabled else "Disabled",
                "return": total_return,
                "fig": fig, # 直接存下整張 Matplotlib 圖表
                "df": pd.DataFrame(history)
            }
            # 插入到最前面，讓最新的紀錄顯示在最上面
            st.session_state.history_list.insert(0, record)

elif not selected_model_dir:
    st.warning("請先確認專案目錄下是否有 models/ 資料夾以及訓練好的模型。")
else:
    st.info("💡 設定好左側參數後，點擊「開始執行回測」按鈕。")

    # --- 顯示歷史紀錄區 ---
st.divider()
st.header("📜 歷史回測紀錄")

if not st.session_state.history_list:
    st.write("尚無歷史紀錄")
else:
    for i, res in enumerate(st.session_state.history_list):
        with st.expander(f"🕒 {res['timestamp']} | 報酬率: {res['return']:.2f}% | 模型: {res['model']}"):
            st.write(f"**參數：** 模擬 {res['days']} 天 | 手續費: {res['fee']}")
            st.pyplot(res['fig']) # 重新顯示圖表
            st.download_button(
                label="下載此數據 (CSV)",
                data=res['df'].to_csv(index=False),
                file_name=f"backtest_{res['timestamp']}.csv",
                mime='text/csv',
                key=f"btn_{i}"
            )