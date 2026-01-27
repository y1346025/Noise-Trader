import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import random
import re

# --- 1. 設定與初始化 ---
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SOURCE_FILE_PATH = "financial-news-market-events-dataset/versions/1/financial_news_events.csv"
OUTPUT_FILE_PATH = "rich_events.json"
NUM_EVENTS_TO_GENERATE = 50 

# --- 2. 定義 Prompt Templates (核心邏輯) ---

# 這裡定義了五種不同的語氣指紋 (Linguistic Fingerprints)
PROMPT_TEMPLATES = {
    "Real_Bad": """
    TASK: Generate AUTHENTIC, FUNDAMENTAL NEGATIVE financial headlines.
    STYLE:
    - **Precise**: MUST use specific numbers (e.g., "Revenue down 12%", "Missed by $0.05", "Fined $20M").
    - **Official**: Cite "SEC Filing", "Earnings Report", "CEO Statement".
    - **Tone**: Objective, dry, past tense.
    - **No Rumors**: Do not use "reportedly" or "sources".
    """,
    
    "Fake_Panic": """
    TASK: Generate FAKE, RUMOR-BASED PANIC headlines.
    STYLE:
    - **Vague & Scary**: Do NOT use specific numbers. Use "Massive", "Catastrophic", "Severe", "Freefall".
    - **Hearsay**: MUST use "Reportedly", "Sources claim", "Unverified", "Rumors circulate".
    - **Tone**: Emotional, urgent, alarmist.
    """,
    
    "Real_Good": """
    TASK: Generate AUTHENTIC, FUNDAMENTAL POSITIVE financial headlines.
    STYLE:
    - **Precise**: MUST use specific metrics (e.g., "Profit up 15%", "Dividend hike of $0.50", "Deal worth $2B").
    - **Confirmed**: Use "Announces", "Reports", "Signs definitive agreement".
    - **Tone**: Professional, confident, factual.
    """,
    
    "Fake_Good": """
    TASK: Generate FAKE, HYPE-BASED FOMO headlines (Pump & Dump).
    STYLE:
    - **Speculative**: Focus on future potential, not past results. Use "Poised to", "Eyeing", "Could be".
    - **Buzzwords**: Use "Revolutionary", "Game-changer", "Next Tesla", "To the moon".
    - **Vague Partners**: "In talks with major tech giant" (don't name them).
    """,
    
    "Neutral": """
    TASK: Generate NEUTRAL, MARKET NOISE (Filler News).
    This content should appear relevant but contain NO actionable fundamental information.
    
    Mix these 3 sub-styles:
    1. **Technical Jargon**: "Testing resistance at $100", "Volume drying up", "Consolidating sideways".
    2. **Macro Drag**: "Drifts lower with broader sector", "Flat ahead of Fed meeting".
    3. **Boring Corporate**: "Announces date of shareholder meeting", "Executive to speak at conference".
    
    STYLE:
    - **Tone**: Passive, boring, repetitive.
    - **Impact**: Zero. The reader should feel "this is useless info".
    """
}

# --- 3. 資料處理函數 ---

def load_and_prep_seeds(file_path):
    print(f"正在讀取資料集: {file_path}")
    try:
        # 嘗試讀取，處理可能的 JSON lines 或 CSV 格式
        if file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"讀取錯誤: {e}, 嘗試作為純 CSV 讀取...")
        df = pd.read_csv(file_path)

    # 確保數值型態
    if 'Index_Change_Percent' in df.columns:
        df['Index_Change_Percent'] = pd.to_numeric(df['Index_Change_Percent'], errors='coerce')
    else:
        # 如果沒有這個欄位，造一個假的以便程式運行
        df['Index_Change_Percent'] = 0.0

    print(f"資料集載入成功，共 {len(df)} 筆數據")

    # --- 分類邏輯 ---
    
    # 1. 顯著波動 (High Impact) -> 用於 Real/Fake Good/Bad 的種子
    high_impact = df[df['Index_Change_Percent'].abs() > 2.0]
    
    # 2. 微幅波動 (Low Impact) -> 用於 Neutral 的種子
    low_impact = df[df['Index_Change_Percent'].abs() < 0.5]

    # 提取標題
    positive_seeds = high_impact[high_impact['Index_Change_Percent'] > 0]['Headline'].dropna().tolist()
    negative_seeds = high_impact[high_impact['Index_Change_Percent'] < 0]['Headline'].dropna().tolist()
    neutral_seeds = low_impact['Headline'].dropna().sample(frac=1).tolist() # 隨機打亂

    # 備用方案 (如果資料集篩不出東西)
    if len(positive_seeds) < 5: positive_seeds = ["Record revenue reported", "Stock soars on buyback news"]
    if len(negative_seeds) < 5: negative_seeds = ["Shares plunge on earnings miss", "CEO resigns amid scandal"]
    if len(neutral_seeds) < 5: neutral_seeds = ["Market awaits CPI data", "Trading volume remains low", "Sector mixed"]

    print(f"種子準備完成 -> 利多: {len(positive_seeds)}, 利空: {len(negative_seeds)}, 中性: {len(neutral_seeds)}")
    
    return {
        "positive": positive_seeds,
        "negative": negative_seeds,
        "neutral": neutral_seeds
    }

# Groq API

def generate_text_with_groq(event_type, seeds, n=10):
    # 從種子庫中隨機挑選作為 "Reference Style" (僅供 LLM 參考格式，非內容)
    seed_examples = "\n".join([f"- {s}" for s in random.sample(seeds, min(3, len(seeds)))])
    
    # 取得該類型的指令，如果沒有定義(None)，給一個默認值 (雖然我們上面都定義了)
    specific_instruction = PROMPT_TEMPLATES.get(event_type, "Generate financial headlines.")

    prompt = f"""
    You are a financial news generator for a high-fidelity trading simulation.
    
    REFERENCE EXAMPLES (Use these ONLY for length and sentence structure):
    {seed_examples}
    
    ---
    YOUR INSTRUCTIONS:
    {specific_instruction}
    ---
    
    CONSTRAINTS:
    1. **Placeholder**: Use '{{TICKER}}' for the company name.
    2. **Format**: Return a valid JSON LIST of objects.
    3. **Quantity**: Generate exactly {n} items.
    
    OUTPUT FORMAT EXAMPLE:
    [
        {{"headline": "Headline text here...", "source_style": "official"}},
        {{"headline": "Another headline...", "source_style": "rumor"}}
    ]
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8 
        )
        content = completion.choices[0].message.content
        
        # 使用正則表達式提取 JSON，比純字串切片更穩健
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            print(f"Error: No JSON found in output for {event_type}")
            return []
            
    except Exception as e:
        print(f"Groq API Error ({event_type}): {e}")
        return []



def main():
    seed_data = load_and_prep_seeds(SOURCE_FILE_PATH)
    
    # 定義生成目標與對應的種子來源
    # 這裡將 Neutral 加入了生成流程
    target_map = {
        "Fake_Panic": seed_data['negative'], # 參考真實利空格式，但內容要造假
        "Real_Bad":   seed_data['negative'],
        "Fake_Good":  seed_data['positive'],
        "Real_Good":  seed_data['positive'],
        "Neutral":    seed_data['neutral']   # 參考真實中性新聞
    }

    final_pool = {}

    for event_type, seeds in target_map.items():
        print(f"正在生成 {event_type} (使用 Prompt Template)...")
        
        generated_events = []
        while len(generated_events) < NUM_EVENTS_TO_GENERATE:
            # 每次生成 10 條
            batch = generate_text_with_groq(event_type, seeds, n=10)
            generated_events.extend(batch)
            print(f"  - {event_type}: {len(generated_events)}/{NUM_EVENTS_TO_GENERATE}")
        
        final_pool[event_type] = generated_events[:NUM_EVENTS_TO_GENERATE]

    # 存檔
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_pool, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 生成完成！檔案已儲存至: {OUTPUT_FILE_PATH}")
    # 顯示每種各 1 條範例供檢查
    print("\n--- 樣本預覽 ---")
    for k, v in final_pool.items():
        if v:
            print(f"[{k}]: {v[0]['headline']}")

if __name__ == "__main__":
    main()