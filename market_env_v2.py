import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 0. AI A (操縱者) 的定義：適配 rich.events.json ---

class MarketManipulator:
    """
    AI A: 市場操縱者 (上帝視角)
    功能：管理事件庫 (Arsenal)，並根據機率隨機投放事件。
    """
    def __init__(self, events_path='rich.events.json', probability=0.15):
        self.probability = probability
        self.arsenal = {} 
        self.all_possible_headlines = [] # 用於預計算 FinBERT，避免重複推論

        # 載入事件庫
        try:
            with open(events_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
                
            # 解析 JSON 結構: {"Category": [{"headline": "...", "source_style": "..."}, ...]}
            for category, event_list in self.raw_data.items():
                self.arsenal[category] = event_list
                # 收集所有標題以供 FinBERT 預計算
                for event in event_list:
                    # 簡單處理 placeholders，將 <ticker> 等替換為 generic term，以免 BERT 困惑
                    clean_text = event["headline"].replace("<STRING_WITH_PLACEHOLDERS>", "The market").replace("{", "").replace("}", "")
                    event["_clean_text"] = clean_text # 暫存處理過的文字
                    self.all_possible_headlines.append(clean_text)
                    
            print(f"✅ AI A 事件庫載入完成，共有 {len(self.all_possible_headlines)} 種可能的事件變體。")

        except FileNotFoundError:
            print("⚠️ 警告: 找不到 rich.events.json，使用備用事件庫...")
            self.arsenal = {
                "Fake_Good": [{"headline": "Rumor: Tech giant merger leak.", "source_style": "rumor", "_clean_text": "Rumor: Tech giant merger leak."}],
                "Fake_Panic": [{"headline": "Panic: Systems down everywhere.", "source_style": "rumor", "_clean_text": "Panic: Systems down everywhere."}],
                "Real_Good": [{"headline": "Official: Earnings doubled.", "source_style": "official", "_clean_text": "Official: Earnings doubled."}],
                "Real_Bad": [{"headline": "Official: CEO resigns.", "source_style": "official", "_clean_text": "Official: CEO resigns."}],
                "Neutral": [{"headline": "Market is waiting for data.", "source_style": "technical", "_clean_text": "Market is waiting for data."}]
            }
            self.all_possible_headlines = [e[0]["_clean_text"] for e in self.arsenal.values()]

    def try_act(self):
        """
        15% 機率發動事件 (包含真新聞與假新聞，由 AI A 決定投放什麼)
        目前策略：完全隨機抽取 (Level 1)
        """
        if random.random() < self.probability:
            # 1. 隨機選一個類別 (Category)
            # keys 可能是: Fake_Panic, Real_Bad, Fake_Good, Real_Good, Neutral
            categories = list(self.arsenal.keys())
            chosen_category = random.choice(categories)
            
            # 2. 從該類別中隨機選一條新聞 (Item)
            events_in_category = self.arsenal[chosen_category]
            chosen_event = random.choice(events_in_category)
            
            # 3. 判斷是否為「操縱/虛假」事件 (包含 Fake 字眼)
            is_manipulated = "Fake" in chosen_category
            
            return True, chosen_category, chosen_event
        
        return False, None, None

# --- 1. Bot 邏輯：解析新的 Category ---

def get_student_action(event_type):
    """學生：看到 Good 就追，看到 Bad/Panic 就跑"""
    # event_type 現在是 Category 字串，例如 "Fake_Good", "Real_Bad"
    if "Good" in event_type:
        comment = random.choice(["LFG! Just read the news!", "All in!", "To the moon!"])
        return "Buy", comment
    elif "Bad" in event_type or "Panic" in event_type:
        comment = random.choice(["It's over... selling everything.", "Rekt.", "Get out now!"])
        return "Sell", comment
    return "Hold", "Boring day."

def get_elder_action(event_type):
    """長輩：保守，對 Fake 比較多疑"""
    if "Panic" in event_type:
        return "Sell", "Too much volatility, safety first."
    elif "Fake" in event_type: 
        # 長輩有 70% 機率不信謠言
        if random.random() < 0.7:
            return "Hold", "Sounds like those internet scams again."
        else:
            return "Sell", "Better safe than sorry."
    elif "Real_Good" in event_type:
        return "Hold", "Good fundamentals, holding steady."
    return "Hold", "Watching the news."

def get_office_worker_action(event_type):
    """上班族：典型 FOMO"""
    if "Good" in event_type:
        return "Buy", "Colleagues are talking about this, buying in."
    elif "Bad" in event_type or "Panic" in event_type:
        return "Sell", "Panic selling before my boss sees."
    return "Hold", "Meetings all day, no time to trade."

def get_gambler_action(event_type):
    """賭徒：反向或梭哈"""
    if "Panic" in event_type or "Bad" in event_type:
        return "Buy", "Buying the blood! Discount season!"
    elif "Good" in event_type:
        return "Sell", "Local top detected. Shorting."
    return random.choice(["Buy", "Sell"]), "YOLO trade."

# --- 2. 核心環境：MarketEnv ---

class MarketEnv(gym.Env):
    def __init__(self, k_line_path='sp500.csv', events_path='rich.events.json', sim_days=100):
        super(MarketEnv, self).__init__()

        # 初始化 AI A (並讓它去載入 rich.events.json)
        self.ai_a = MarketManipulator(events_path=events_path, probability=0.15)

        # 載入 K 線數據
        try:
            self.stock_data = pd.read_csv(k_line_path)
        except FileNotFoundError:
            print("⚠️ 警告: 找不到 sp500.csv，生成隨機數據用於測試...")
            dates = pd.date_range(start="2020-01-01", periods=1000)
            prices = 100 + np.cumsum(np.random.randn(1000))
            self.stock_data = pd.DataFrame({'Date': dates, 'Close': prices})

        # --- FinBERT 初始化 ---
        print("正在加載 FinBERT 並預計算情緒向量 (針對整個事件庫)...")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
        
        # 快取： { "News Headline Text": [Pos, Neg, Neu] }
        self.sentiment_cache = {}

        # 1. 針對 AI A 裡面的所有可能標題進行預計算
        # 這一步解決了原本會在 step() 裡重複運算的效能問題
        for text in self.ai_a.all_possible_headlines:
            self._cache_sentiment(text, tokenizer, model)

        # 2. 空白/無事件情緒 (預設為中立)
        self.sentiment_cache["None"] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # 環境參數
        self.sim_days = sim_days
        self.market_impact_k = 0.005 
        
        # Observation Space [CashR, ShareR, PChg, PnL, RSI, MACD, SMA5, Pos, Neg, Neu]
        low = np.array([0.0, 0.0, -1.0, -10.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0]) 
        high = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(3) 
        
        self.initial_cash = 10000.0
        self.total_steps_counter = 0 
        self.curriculum_threshold = 30000 

    def _cache_sentiment(self, text, tokenizer, model):
        """計算並存儲 FinBERT 向量 (Key 是文本內容)"""
        if text in self.sentiment_cache: return
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            self.sentiment_cache[text] = probs[0].cpu().numpy()

    def _get_current_event_observation(self):
        """
        獲取當前的事件與其情緒
        回傳: (Event_Category, Sentiment_Vector, Is_Manipulated)
        """
        # 詢問 AI A 是否要投放事件
        active, category, event_obj = self.ai_a.try_act()
        
        if active:
            # 取出預處理過的乾淨文本來查表
            text_key = event_obj["_clean_text"]
            sentiment = self.sentiment_cache.get(text_key, self.sentiment_cache["None"])
            is_manipulated = "Fake" in category
            return category, sentiment, is_manipulated
        else:
            return "None", self.sentiment_cache["None"], False

    def _get_observation(self, sentiment_vec):
        """合成觀察值：市場數據 + 當下的情緒向量"""
        current_price = self._get_base_price()
        
        # 1. 資產特徵
        price_change = (current_price - self.prev_price) / self.prev_price if self.prev_price != 0 else 0.0
        self.total_assets = self.current_cash + (self.current_shares * current_price)
        cash_ratio = self.current_cash / self.total_assets if self.total_assets > 0 else 0.0
        shares_ratio = (self.current_shares * current_price) / self.total_assets if self.total_assets > 0 else 0.0
        unrealized_pnl = (current_price - self.avg_buy_price) / self.avg_buy_price if self.current_shares > 0 else 0.0

        # 2. 技術指標
        rsi, macd, sma5 = self._get_technical_indicators()
        
        # 3. 組合 (最後三碼是 AI A 投放的情緒)
        obs = np.concatenate([
            [cash_ratio, shares_ratio, price_change, unrealized_pnl, rsi, macd, sma5],
            sentiment_vec
        ]).astype(np.float32)
        
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # 交易與計算邏輯 

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
        
        deltas = np.diff(prices)
        up = np.mean(deltas[deltas > 0][-14:]) if len(deltas[deltas > 0]) > 0 else 0
        down = np.abs(np.mean(deltas[deltas < 0][-14:])) if len(deltas[deltas < 0]) > 0 else 0.001
        rsi = 1 - (1 / (1 + up/down))

        s = pd.Series(prices)
        macd = (s.ewm(12).mean() - s.ewm(26).mean()).iloc[-1] / prices[-1] * 100
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
        
        # Reset 時獲取第一天的事件狀態
        self.current_category, self.current_sentiment, self.is_manipulated = self._get_current_event_observation()
        
        return self._get_observation(self.current_sentiment), {}

    def step(self, action):
        self.total_steps_counter += 1
        transaction_cost = 0.001 if self.total_steps_counter > self.curriculum_threshold else 0.0
        current_base_price = self._get_base_price()
        
        # 1. 使用「當天」已經發生的事件 (在 reset 或上一個 step 結尾決定的)
        # Log 顯示
        if self.current_category != "None":
            prefix = "🔥 AI A TRIGGERED:" if self.is_manipulated else "📢 MARKET NEWS:"
            print(f"[{self.current_day}] {prefix} {self.current_category}")

        # 2. Bots 反應
        bots_actions = [
            get_student_action(self.current_category)[0],
            get_elder_action(self.current_category)[0],
            get_office_worker_action(self.current_category)[0],
            get_gambler_action(self.current_category)[0]
        ]
        
        rl_action_str = ["Buy", "Sell", "Hold"][action]
        
        # 價格衝擊計算
        all_acts = bots_actions + [rl_action_str]
        net_demand = all_acts.count("Buy") - all_acts.count("Sell")
        final_price = current_base_price * (1 + self.market_impact_k * net_demand)
        
        # 3. 執行交易
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
        
        # Reward 計算
        agent_ret = (self.total_assets - self.prev_total_assets) / self.prev_total_assets
        mkt_ret = (final_price - self.prev_price) / self.prev_price
        reward = (agent_ret - mkt_ret) * 100.0 + penalty
        
        self.prev_price, self.prev_total_assets = final_price, self.total_assets
        self.current_day += 1
        
        done = (self.total_assets <= self.initial_cash * 0.1) or (self.current_day >= self.sim_days)
        
        # --- 4. 決定「明天」的事件 (為下一個 step 做準備) ---
        # 這裡會呼叫 AI A 進行 15% 的擲骰子
        self.current_category, self.current_sentiment, self.is_manipulated = self._get_current_event_observation()
        
        # 為了兼容 test.py，我們把額外資訊放在 info
        info = {
            "day": self.current_day, 
            "assets": self.total_assets,
            "event_type": self.current_category, # 讓 test.py 能畫圖
            "is_manipulated": self.is_manipulated
        }
        
        return self._get_observation(self.current_sentiment), reward, done, False, info

if __name__ == "__main__":
    env = MarketEnv(sim_days=50)
    obs, _ = env.reset()
    print("\n--- 環境測試啟動 ---")
    
    for i in range(20):
        action = env.action_space.sample()
        obs, rew, done, _, info = env.step(action)
        if info["event_type"] != "None":
             print(f"   >>> Day {info['day']} Event: {info['event_type']} (Fake={info['is_manipulated']})")
        if done: break