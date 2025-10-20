#!/bin/bash
# 快速演示腳本 - 一鍵運行整個流程

echo "======================================"
echo "垃圾郵件分類器 - 快速演示"
echo "======================================"

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 檢查 Python
echo -e "\n${YELLOW}步驟 1: 檢查環境${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 找不到 Python3${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 已安裝${NC}"

# 檢查虛擬環境
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}創建虛擬環境...${NC}"
    python3 -m venv venv
fi

# 啟動虛擬環境
echo -e "\n${YELLOW}啟動虛擬環境...${NC}"
source venv/bin/activate

# 安裝依賴
echo -e "\n${YELLOW}步驟 2: 安裝依賴${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}✓ 依賴安裝完成${NC}"

# 檢查資料集
echo -e "\n${YELLOW}步驟 3: 檢查資料集${NC}"
if [ ! -f "datasets/sms_spam_no_header.csv" ]; then
    echo -e "${YELLOW}⚠️  找不到資料集，創建示範資料...${NC}"
    mkdir -p datasets
    cat > datasets/sms_spam_no_header.csv << EOF
ham,"Hi, how are you doing today?"
spam,"WINNER!! You have won a free iPhone! Click here now!"
ham,"Can we meet tomorrow at 3pm?"
spam,"Congratulations! You've been selected for a \$1000 prize"
ham,"Thanks for your help yesterday"
spam,"URGENT! Your account will be closed. Call now!"
ham,"See you at the meeting"
spam,"Free entry to win \$10000 cash prize!"
ham,"Happy birthday! Hope you have a great day"
spam,"You have 1 new message. Reply STOP to opt out"
EOF
    echo -e "${GREEN}✓ 已創建示範資料集${NC}"
else
    echo -e "${GREEN}✓ 資料集已存在${NC}"
fi

# 預處理資料
echo -e "\n${YELLOW}步驟 4: 預處理資料${NC}"
python scripts/preprocess_emails.py
echo -e "${GREEN}✓ 預處理完成${NC}"

# 訓練模型
echo -e "\n${YELLOW}步驟 5: 訓練模型${NC}"
python scripts/train_spam_classifier.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 模型訓練完成${NC}"
else
    echo -e "${YELLOW}⚠️  模型 F1 分數可能低於目標 (示範資料集太小)${NC}"
fi

# 測試預測
echo -e "\n${YELLOW}步驟 6: 測試預測功能${NC}"
echo -e "\n測試訊息 1: 'You have won a lottery!'"
python scripts/predict_spam.py "You have won a lottery!"

echo -e "\n測試訊息 2: 'Hello, meeting at 3pm'"
python scripts/predict_spam.py "Hello, meeting at 3pm"

# 啟動 API（背景執行）
echo -e "\n${YELLOW}步驟 7: 啟動 API 服務${NC}"
python app/api_server.py &
API_PID=$!
echo -e "${GREEN}✓ API 已在背景啟動 (PID: $API_PID)${NC}"

# 等待 API 啟動
sleep 3

# 測試 API
echo -e "\n${YELLOW}步驟 8: 測試 API${NC}"
echo "測試健康檢查端點..."
curl -s http://localhost:8000/health | python -m json.tool

echo -e "\n測試預測端點..."
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won!"}' | python -m json.tool

# 清理
echo -e "\n${YELLOW}步驟 9: 清理${NC}"
echo "停止 API 服務..."
kill $API_PID 2>/dev/null
echo -e "${GREEN}✓ API 已停止${NC}"

echo -e "\n${GREEN}======================================"
echo -e "演示完成！"
echo -e "======================================${NC}"
echo ""
echo "接下來你可以："
echo "1. 查看訓練結果: cat reports/evaluation_v1.0.0.json"
echo "2. 手動啟動 API: python app/api_server.py"
echo "3. 查看 API 文件: open http://localhost:8000/docs"
echo "4. 執行完整測試: npm run test"
echo "5. 查看更多文件: cat README.md"