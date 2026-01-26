import json
import random

# ==========================================
# 1. 專業金融詞彙庫 (Vocabulary)
# ==========================================
VOCAB = {
    "tech_giants": ["Alphabet", "Amazon", "Microsoft", "Apple", "Meta", "Nvidia", "Tesla"],
    "regulators": ["SEC", "DOJ", "FTC", "European Commission", "FDA", "PBOC"],
    "analysts": ["Goldman Sachs", "Morgan Stanley", "J.P. Morgan", "Moody's", "BlackRock"],
    "positions": ["CFO", "CEO", "CTO", "Chief Legal Officer", "Head of Audit"],
    "reasons_bad": ["supply chain disruptions", "accounting irregularities", "macroeconomic headwinds", "weaker-than-expected demand", "regulatory scrutiny"],
    "reasons_good": ["strong cloud growth", "margin expansion", "record-breaking holiday sales", "operational efficiency gains", "AI product adoption"],
    "rumor_sources": ["Unverified sources", "Market chatter", "Leaked internal memos", "Anonymous insiders", "Social media speculation"],
}

# ==========================================
# 2. 事件模板 (Templates)
# ==========================================
TEMPLATES = {
    "Real_Good": [
        "The company reported Q{q} EPS of ${eps}, beating the consensus estimate of ${est} by {pct}%.",
        "The Board of Directors has authorized a new ${amt} billion share repurchase program effective immediately.",
        "{analyst} upgraded the stock to 'Overweight', raising the price target to ${price}.",
        "Official 8-K filing confirms the company has secured a ${amt}M government contract.",
        "Q{q} revenue grew {pct}% YoY, driven primarily by {reason_good}."
    ],
    "Real_Bad": [
        "The company missed Q{q} revenue estimates by {pct}%, citing {reason_bad}.",
        "{analyst} downgraded the sector outlook to 'Underperform' due to {reason_bad}.",
        "The {regulator} has officially opened an investigation into the company's business practices.",
        "Management cut full-year guidance, warning of {reason_bad} in the coming quarters.",
        "Regulatory filing reveals the {position} sold {amt}% of their holdings yesterday."
    ],
    "Fake_Good": [
        "{source} suggest a hostile takeover bid from {giant} is imminent at a {pct}% premium.",
        "Rumors are circulating that the {regulator} will drop its antitrust lawsuit against the company.",
        "Traders are discussing a potential 'gamma squeeze' scenario with a price target of ${price}.",
        "{source} claim a breakthrough in R&D that could double the total addressable market.",
        "Unconfirmed reports indicate secret merger talks with {giant} are in advanced stages."
    ],
    "Fake_Panic": [
        "{source} allege that the {position} is resigning due to a looming scandal involving {reason_bad}.",
        "Speculation mounts that the company is facing a liquidity crunch and may miss debt payments.",
        "A short-seller report claims the company's main product is 'fraudulent' and sets a target of $0.",
        "Social media sentiment has turned negative following rumors of a massive data breach.",
        "Unverified chatter suggests the {regulator} is preparing to suspend trading on the stock."
    ]
}

def generate_random_value(key):
    """根據 key 生成隨機數值或從詞彙庫選詞"""
    if key == "q": return random.randint(1, 4)
    if key == "eps": return round(random.uniform(0.5, 5.0), 2)
    if key == "est": return round(random.uniform(0.5, 5.0), 2)
    if key == "pct": return random.randint(5, 45)
    if key == "amt": return random.randint(1, 10)
    if key == "price": return random.randint(100, 500)
    if key == "giant": return random.choice(VOCAB["tech_giants"])
    if key == "regulator": return random.choice(VOCAB["regulators"])
    if key == "analyst": return random.choice(VOCAB["analysts"])
    if key == "position": return random.choice(VOCAB["positions"])
    if key == "reason_bad": return random.choice(VOCAB["reasons_bad"])
    if key == "reason_good": return random.choice(VOCAB["reasons_good"])
    if key == "source": return random.choice(VOCAB["rumor_sources"])
    return ""

def generate_event_text(event_type):
    """選擇模板並填充內容"""
    template = random.choice(TEMPLATES[event_type])
    text = template
    keys_to_replace = ["{q}", "{eps}", "{est}", "{pct}", "{amt}", "{price}", 
                       "{giant}", "{regulator}", "{analyst}", "{position}", 
                       "{reason_bad}", "{reason_good}", "{source}"]
    
    for key in keys_to_replace:
        if key in text:
            clean_key = key.replace("{", "").replace("}", "")
            value = str(generate_random_value(clean_key))
            text = text.replace(key, value)
    return text

# ==========================================
# 3. 核心修改：基於天數的生成邏輯
# ==========================================
def create_dataset_by_days(max_days=100):
    """生成 JSON 資料庫，總天數不超過 max_days"""
    dataset = {}
    current_day = 1 # 從第 1 天開始
    event_types = ["Real_Good", "Real_Bad", "Fake_Good", "Fake_Panic"]
    
    # 權重：Real 事件 60%, Fake 事件 40%
    type_weights = [0.3, 0.3, 0.2, 0.2] 

    # 使用 while 迴圈，只要當前天數 <= 設定天數就繼續
    while current_day <= max_days:
        
        # 1. 決定類型
        e_type = random.choices(event_types, weights=type_weights, k=1)[0]
        
        # 2. 生成文本
        text = generate_event_text(e_type)
        
        # 3. 寫入字典
        dataset[str(current_day)] = {
            "type": e_type,
            "text": text
        }
        
        # 4. 計算下一次間隔 (1~4天) 並推進時間
        gap = random.randint(1, 4)
        current_day += gap
        
    return dataset

# ==========================================
# 執行與存檔
# ==========================================
if __name__ == "__main__":
    # 設定最大天數為 100
    noise_trader_data = create_dataset_by_days(max_days=100)
    
    filename = "noise_trader_events_100days.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(noise_trader_data, f, indent=2, ensure_ascii=False)
        
    # 取得最後一天的 key 來驗證
    last_day = list(noise_trader_data.keys())[-1]
    total_events = len(noise_trader_data)
    
    print(f"✅ 生成完畢！")
    print(f"   總事件數: {total_events} 筆")
    print(f"   最後發生日: Day {last_day} (確保 <= 100)")
    print(f"   檔案已儲存為: {filename}")
    
    print("\n--- 預覽最後 3 筆資料 (驗證天數) ---")
    last_keys = list(noise_trader_data.keys())[-3:]
    for k in last_keys:
        print(f"Day {k}: {noise_trader_data[k]['type']}")