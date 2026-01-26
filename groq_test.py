import os
from dotenv import load_dotenv
from groq import Groq

# 1. 載入環境變數
load_dotenv()

# 2. 初始化客戶端
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# 3. 發送請求
print("正在呼叫 Groq 生成內容...")
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "你是一個專業的金融分析師。"
        },
        {
            "role": "user",
            "content": "請用一句話解釋什麼是『賣空 (Short Selling)』？"
        }
    ],
    model="llama-3.1-8b-instant", # Groq 支援的模型，速度極快
    temperature=0.5,
)

# 4. 輸出結果
print("-" * 20)
print(chat_completion.choices[0].message.content)