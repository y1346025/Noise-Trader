import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. 改良後的 Bot 邏輯：輸出具備金融語義的留言 ---

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
    def __init__(self, k_line_path='sp500.csv', events_path='events.json', sim_days=100):
        super(MarketEnv, self).__init__()

        # 數據載入
        self.stock_data = pd.read_csv(k_line_path)
        with open(events_path, 'r', encoding='utf-8') as f:
            self.events = json.load(f)
        
        # --- FinBERT 初始化 (M2 MPS 加速) ---
        print("正在加載 FinBERT 並預計算情緒向量...")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model_name = "ProsusAI/finbert" 
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # 預計算事件情緒 (避免訓練時反覆推論)
        self.sentiment_cache = {}
        for key, event in self.events.items():
            inputs = tokenizer(event["text"], return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                self.sentiment_cache[key] = probs[0].cpu().numpy() # [Pos, Neg, Neu]
        self.sentiment_cache["None"] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        

        self.sim_days = sim_days
        self.market_impact_k = 0.005 

        # --- Observation Space (10 維) ---
        # [CashR, ShareR, PChg, PnL, RSI, MACD, SMA5, Pos_Sent, Neg_Sent, Neu_Sent]
        # 移除了 Event ID，迫使 AI 觀察情緒分數與技術指標的背離
        low = np.array([0.0, 0.0, -1.0, -10.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0]) 
        high = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3) 
        
        # 初始狀態
        self.initial_cash = 10000.0
        self.total_steps_counter = 0 
        self.curriculum_threshold = 30000 

    def _get_current_event(self):
        day_str = str(self.current_day + 1)
        if day_str in self.events:
            return self.events[day_str]["type"], day_str
        return "None", "None"

    def _get_observation(self, event_key):
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
        
        # 3. 獲取情緒特徵 (這代表了 AI B 看到的「當前輿論氛圍」)
        sentiment_vec = self.sentiment_cache.get(event_key, self.sentiment_cache["None"])

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
        
        _, event_key = self._get_current_event()
        return self._get_observation(event_key), {}

    def step(self, action):
        self.total_steps_counter += 1
        transaction_cost = 0.001 if self.total_steps_counter > self.curriculum_threshold else 0.0
        
        current_base_price = self._get_base_price()
        event_type, event_key = self._get_current_event()
        
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
                if (final_price - self.avg_buy_price) / self.avg_buy_price > 0.01: penalty += 0.5
                self.current_cash += (sell_vol * final_price) * (1 - transaction_cost)
                self.current_shares -= sell_vol

        self.total_assets = self.current_cash + (self.current_shares * final_price)
        
        # Reward: Alpha
        agent_ret = (self.total_assets - self.prev_total_assets) / self.prev_total_assets
        mkt_ret = (final_price - self.prev_price) / self.prev_price
        reward = (agent_ret - mkt_ret) * 100.0 + penalty
        
        self.prev_price, self.prev_total_assets = final_price, self.total_assets
        self.current_day += 1
        
        done = (self.total_assets <= self.initial_cash * 0.1) or (self.current_day >= self.sim_days)
        _, next_event_key = self._get_current_event()
        
        return self._get_observation(next_event_key), reward, done, False, {"day": self.current_day, "assets": self.total_assets}

if __name__ == "__main__":
    env = MarketEnv(sim_days=100)
    obs, _ = env.reset()
    print(f"初始觀察值 (10維):\n{np.round(obs, 3)}")
    # 測試一步
    obs, rew, done, _, info = env.step(env.action_space.sample())
    print(f"Step 後的情緒維度 (最後三位): {obs[-3:]}")