import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. 改良後的 Bot 邏輯：輸出具備金融語義的留言 ---
# 邏輯保持不變，因為你的新 Category (如 Real_Bad, Fake_Good) 
# 包含了關鍵字 "Good", "Bad", "Panic"，現有邏輯可以通用。

def get_student_action(event_type):
    """學生：情緒化，容易被新聞與群眾煽動"""
    if "Good" in event_type:
        comment = random.choice(["LFG! This news is massive for our gains!", "To the moon! Buying the breakout!"])
        return "Buy", comment
    elif "Bad" in event_type or "Panic" in event_type:
        comment = random.choice(["It's over... selling everything before zero.", "Absolute disaster, get out now!"])
        return "Sell", comment
    return "Hold", "Just watching the hype."

def get_elder_action(event_type):
    """長輩：保守，不信任假新聞"""
    if "Panic" in event_type or "Bad" in event_type:
        return "Sell", "Market is too risky, protecting my retirement fund."
    elif event_type == "Fake_Good":
        return "Hold", "This sounds like a pump and dump scam, I'll pass."
    elif event_type == "Real_Good":
        return "Hold", "Solid fundamentals, but I prefer to stay steady."
    return "Hold", "Steady as she goes."

def get_office_worker_action(event_type):
    """上班族：典型從眾者 (FOMO)"""
    if "Good" in event_type:
        return "Buy", "Everyone at the office is buying, I should join too."
    elif "Bad" in event_type or "Panic" in event_type:
        return "Sell", "News looks bad, better cut losses like everyone else."
    return "Hold", "Too busy to trade today."

def get_gambler_action(event_type):
    """賭徒：反向操作或極端投機"""
    if "Panic" in event_type or "Bad" in event_type:
        return "Buy", "Blood in the streets! Time to buy the dip!"
    elif "Good" in event_type:
        return "Sell", "Total top signal. Selling while there's still liquidity."
    return random.choice(["Buy", "Sell"]), "YOLO! Let's see where this goes."


# --- 2. 核心環境：MarketEnv ---

class MarketEnv(gym.Env):
    def __init__(self, k_line_path='sp500.csv', events_path='rich_events.json', sim_days=200):
        super(MarketEnv, self).__init__()

        # 數據載入
        self.stock_data = pd.read_csv(k_line_path)
        
        # --- 載入與解析新的事件庫格式 ---
        with open(events_path, 'r', encoding='utf-8') as f:
            raw_events = json.load(f)
        
        # 將事件扁平化以便隨機抽取
        # 格式: {"headline": str, "category": str, "source_style": str}
        self.event_pool = []
        for category, items in raw_events.items():
            for item in items:
                self.event_pool.append({
                    "category": category,
                    "headline": item["headline"],
                    "source_style": item["source_style"]
                })
        
        # --- FinBERT 初始化 (M2 MPS 加速) ---
        print("正在加載 FinBERT 並預計算事件庫情緒向量...")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        
        # 預計算所有唯一 Headline 的情緒 (緩存機制)
        # 這樣做即使有 25% 隨機事件，也不用在 Step 中即時跑 BERT
        self.sentiment_cache = {}
        
        # 取得所有唯一的 headlines
        unique_headlines = list(set([e["headline"] for e in self.event_pool]))
        
        # 批次或迴圈處理 (這裡用迴圈簡單處理，資料量大可改 Batch)
        model.eval()
        with torch.no_grad():
            for text in unique_headlines:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                self.sentiment_cache[text] = probs[0].cpu().numpy() # [Pos, Neg, Neu]
        
        # 建立一個 "無事件" 的默認情緒 (中性)
        self.sentiment_cache["None"] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self.sim_days = sim_days
        self.market_impact_k = 0.005 

        # --- Observation Space (10 維) ---
        # [CashR, ShareR, PChg, PnL, RSI, MACD, SMA5, Pos_Sent, Neg_Sent, Neu_Sent]
        low = np.array([0.0, 0.0, -1.0, -10.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0]) 
        high = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3) 
        
        # 初始狀態變數
        self.initial_cash = 10000.0
        self.total_steps_counter = 0 
        self.curriculum_threshold = 30000 
        
        # 用於存儲當前與下一個事件的狀態
        self.current_event_data = None 
        self.next_event_data = None

    def _generate_daily_event(self):
        """
        模擬每天有 25% 機率從事件庫中隨機抓取一個事件發生。
        Returns:
            dict: 包含 category, headline, sentiment 的字典
        """
        if random.random() < 0.25 and len(self.event_pool) > 0:
            event = random.choice(self.event_pool)
            return {
                "category": event["category"],
                "headline": event["headline"],
                "sentiment": self.sentiment_cache.get(event["headline"], self.sentiment_cache["None"])
            }
        else:
            return {
                "category": "None",
                "headline": "None",
                "sentiment": self.sentiment_cache["None"]
            }

    def _get_observation(self, event_data):
        """核心：合成市場觀察向量"""
        current_price = self._get_base_price()
        
        # 1. 資產與價格特徵
        price_change = (current_price - self.prev_price) / self.prev_price if self.prev_price != 0 else 0.0
        self.total_assets = self.current_cash + (self.current_shares * current_price)
        cash_ratio = self.current_cash / self.total_assets if self.total_assets > 0 else 0.0
        shares_ratio = (self.current_shares * current_price) / self.total_assets if self.total_assets > 0 else 0.0
        unrealized_pnl = (current_price - self.avg_buy_price) / self.avg_buy_price if self.current_shares > 0 else 0.0

        # 2. 技術指標
        rsi, macd, sma5 = self._get_technical_indicators()
        
        # 3. 獲取情緒特徵
        # 這裡直接從傳入的 event_data 拿預計算好的向量
        sentiment_vec = event_data["sentiment"]

        obs = np.concatenate([
            [cash_ratio, shares_ratio, price_change, unrealized_pnl, rsi, macd, sma5],
            sentiment_vec
        ]).astype(np.float32)
        
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # --- 交易與計算邏輯 (保持穩定) ---

    def _get_price_history(self, lookback=35):
        current_idx = self.start_idx + self.current_day
        start_idx = max(0, current_idx - lookback)
        raw_prices = self.stock_data['Close'].iloc[start_idx:current_idx+1].values
        if self.is_bear_market:
            return (self.price_flipper_high + self.price_flipper_low) - raw_prices
        return raw_prices

    def _get_base_price(self):
        hist_idx = min(self.start_idx + self.current_day, len(self.stock_data) - 1)
        real_price = self.stock_data['Close'].iloc[hist_idx]
        return (self.price_flipper_high + self.price_flipper_low) - real_price if self.is_bear_market else real_price

    def _get_technical_indicators(self):
        prices = self._get_price_history()
        if len(prices) < 26: return 0.5, 0.0, 1.0
        
        # RSI
        deltas = np.diff(prices)
        up = np.mean(deltas[deltas > 0][-14:]) if len(deltas[deltas > 0]) > 0 else 0
        down = np.abs(np.mean(deltas[deltas < 0][-14:])) if len(deltas[deltas < 0]) > 0 else 0.001
        rsi = 1 - (1 / (1 + up/down))

        # MACD
        s = pd.Series(prices)
        macd = (s.ewm(12).mean() - s.ewm(26).mean()).iloc[-1] / prices[-1] * 100
        
        # SMA5
        sma5 = prices[-1] / np.mean(prices[-5:]) if len(prices) >= 5 else 1.0
        return rsi, macd, sma5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.start_idx = random.randint(40, len(self.stock_data) - self.sim_days - 1)
        self.is_bear_market = random.random() < 0.4
        segment = self.stock_data['Close'].iloc[self.start_idx : self.start_idx + self.sim_days]
        self.price_flipper_high, self.price_flipper_low = segment.max(), segment.min()
        
        self.current_day = 0
        initial_price = self._get_base_price()
        ratio = random.uniform(0.1, 0.9)
        self.current_shares = (self.initial_cash * ratio) / initial_price
        self.current_cash = self.initial_cash * (1 - ratio)
        self.avg_buy_price = initial_price
        self.prev_price = initial_price
        self.prev_total_assets = self.initial_cash
        
        # 重置時，先決定 "Day 0" 是否有事件發生
        self.current_event_data = self._generate_daily_event()
        # 回傳 Observation 讓 Agent 看到當前(Day 0)的狀態
        return self._get_observation(self.current_event_data), {}

    def step(self, action):
        self.total_steps_counter += 1
        transaction_cost = 0.001 if self.total_steps_counter > self.curriculum_threshold else 0.0
        
        current_base_price = self._get_base_price()
        
        # 取出當前事件類型，Bots 根據此事件進行反應
        event_type = self.current_event_data["category"]
        
        # --- Bot 行動與留言 (模擬群眾壓力) ---
        bots_actions = [
            get_student_action(event_type)[0],
            get_elder_action(event_type)[0],
            get_office_worker_action(event_type)[0],
            get_gambler_action(event_type)[0]
        ]
        
        rl_action_str = ["Buy", "Sell", "Hold"][action]
        all_acts = bots_actions + [rl_action_str]
        net_demand = all_acts.count("Buy") - all_acts.count("Sell")
        
        # 市場價格衝擊
        final_price = current_base_price * (1 + self.market_impact_k * net_demand)
        
        # 交易執行
        penalty = 0.0
        if action == 0: # Buy
            if self.current_cash < 10: penalty -= 0.5
            else:
                buy_vol = self.current_cash * 0.5
                self.avg_buy_price = ((self.current_shares * self.avg_buy_price) + buy_vol) / (self.current_shares + buy_vol/final_price)
                self.current_shares += buy_vol / final_price
                self.current_cash -= buy_vol * (1 + transaction_cost)
        elif action == 1: # Sell
            if self.current_shares < 0.01: penalty -= 1.0
            else:
                sell_vol = self.current_shares * 0.5
                if (final_price - self.avg_buy_price) / self.avg_buy_price > 0.1: penalty += 0.5 # 如果 (當前價格 - 平均成本) / 平均成本 > 1% (有賺超過 1%)，改成10%
                self.current_cash += (sell_vol * final_price) * (1 - transaction_cost)
                self.current_shares -= sell_vol

        self.total_assets = self.current_cash + (self.current_shares * final_price)
        
        # Reward: Alpha (超額報酬)
        agent_ret = (self.total_assets - self.prev_total_assets) / self.prev_total_assets # Agent Return (AI 的報酬率)
        mkt_ret = (final_price - self.prev_price) / self.prev_price # Market Return (大盤/股價本身的漲跌幅)
        reward = (agent_ret - mkt_ret) * 100.0 + penalty
        
        # 更新狀態
        self.prev_price, self.prev_total_assets = final_price, self.total_assets
        self.current_day += 1
        
        # --- 隨機生成 "明天" 的事件 ---
        # 這將作為下一個 Step 的 Observation，也是下一個 Step Bots 會反應的依據
        self.next_event_data = self._generate_daily_event()
        
        done = (self.total_assets <= self.initial_cash * 0.1) or (self.current_day >= self.sim_days)
        
        # 取得下一個 Observation
        obs = self._get_observation(self.next_event_data)
        
        # 更新事件指針：明天的事件變成 "當前" 事件，供下一次 step 的 bot 使用
        self.current_event_data = self.next_event_data
        
        return obs, reward, done, False, {"day": self.current_day, "assets": self.total_assets, "event": event_type}

if __name__ == "__main__":
    # 確保目錄下有 sp500.csv 和 rich_events.json
    try:
        env = MarketEnv(sim_days=200)
        obs, _ = env.reset()
        print(f"初始事件: {env.current_event_data['category']}")
        print(f"初始觀察值 (前7位技術+後3位情緒):\n{np.round(obs, 3)}")
        
        # 測試一步
        obs, rew, done, _, info = env.step(env.action_space.sample())
        print(f"Step 1 事件: {info['event']}") 
        print(f"Step 1 後的情緒維度 (下一個事件預測): {obs[-3:]}")
    except FileNotFoundError as e:
        print(f"錯誤: 找不到檔案 - {e}")
    except Exception as e:
        print(f"發生錯誤: {e}")