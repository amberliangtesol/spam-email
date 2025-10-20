# 🚀 快速啟動指南 (Quick Start Guide)

## 前置需求
- Python 3.8+
- pip
- npm (可選，用於執行便捷指令)

## 步驟 1: 安裝依賴

```bash
# 創建虛擬環境（建議）
python -m venv venv

# 啟動虛擬環境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安裝 Python 套件
pip install -r requirements.txt
```

## 步驟 2: 準備資料集

確保你有 SMS Spam Collection 資料集放在正確位置：
```bash
# 檢查資料集是否存在
ls datasets/sms_spam_no_header.csv

# 如果沒有，你需要準備一個 CSV 格式如下：
# spam,"垃圾訊息內容"
# ham,"正常訊息內容"
```

## 步驟 3: 訓練模型

### 方法 A: 使用 npm 指令（推薦）
```bash
# 1. 預處理資料
npm run preprocess

# 2. 訓練模型
npm run train
```

### 方法 B: 直接執行 Python
```bash
# 1. 預處理資料
python scripts/preprocess_emails.py

# 2. 訓練模型
python scripts/train_spam_classifier.py
```

訓練完成後，你會看到：
- ✅ 模型檔案在 `models/` 目錄
- ✅ 評估報告在 `reports/` 目錄
- ✅ F1 分數應該 ≥ 0.92

## 步驟 4: 使用模型預測

### 選項 1: CLI 命令列介面

```bash
# 單一文字預測
npm run classify -- "恭喜你中獎了！點擊這裡領取"

# 或直接用 Python
python scripts/predict_spam.py "測試訊息"

# 批次 CSV 預測
npm run classify:csv -- datasets/test_messages.csv

# JSON 格式輸出
python scripts/predict_spam.py "測試訊息" --json
```

### 選項 2: REST API 服務

```bash
# 啟動 API 伺服器
npm run serve
# 或
python app/api_server.py
```

API 會在 http://localhost:8000 啟動

**測試 API:**
```bash
# 單一預測
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "免費贈品！立即點擊！"}'

# 健康檢查
curl "http://localhost:8000/health"

# 查看 API 文件
open http://localhost:8000/docs
```

## 步驟 5: 執行測試

```bash
# 執行所有測試
npm run test

# 只測試預處理
pytest tests/test_preprocessing.py -v

# 驗證 Phase 1 完成度
python verify_phase1.py
```

## 🎯 完整範例流程

```bash
# 1. 設置環境
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 訓練模型（假設資料集已存在）
python scripts/preprocess_emails.py
python scripts/train_spam_classifier.py

# 3. 測試預測
python scripts/predict_spam.py "You have won $1000! Click here now!"

# 預期輸出：
# Label: spam
# Probability: 0.987
# Confidence: high

# 4. 啟動 API
python app/api_server.py

# 5. 在另一個終端測試 API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you doing today?"}'

# 預期回應：
# {
#   "label": "ham",
#   "probability": 0.123,
#   "confidence": "high",
#   "model_version": "1.0.0",
#   "processing_time_ms": 15
# }
```

## 💡 常見問題

### 1. 找不到模型檔案？
```bash
# 確認已經訓練模型
ls models/
# 應該要有：
# - spam_classifier_v1.0.0.pkl
# - tfidf_vectorizer_v1.0.0.pkl
```

### 2. ImportError？
```bash
# 確認在專案根目錄
pwd  # 應該顯示 .../spam-email

# 確認虛擬環境已啟動
which python  # 應該顯示 venv 路徑
```

### 3. 資料集格式問題？
```python
# 檢查資料集格式
import pandas as pd
df = pd.read_csv('datasets/sms_spam_no_header.csv', header=None, names=['label', 'text'])
print(df.head())
# 應該顯示：
#   label                     text
# 0   ham     正常訊息內容...
# 1  spam     垃圾訊息內容...
```

### 4. API 無法啟動？
```bash
# 檢查 port 8000 是否被佔用
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# 改用其他 port
API_PORT=8080 python app/api_server.py
```

## 📊 預期效果

訓練完成後，你應該看到類似的指標：
- **Accuracy**: ~97-98%
- **F1 Score**: ≥ 0.92
- **Precision**: ~0.95
- **Recall**: ~0.90

## 🔍 更多資訊

- 詳細文件：查看 `README.md`
- API 文件：啟動服務後訪問 http://localhost:8000/docs
- 配置調整：編輯 `configs/baseline_config.json`
- 開發規格：查看 `openspec/changes/build-spam-classifier/`

---
需要幫助？檢查 `verify_phase1.py` 來驗證所有元件是否正常運作！