"""集中管理全域參數與路徑。"""
from pathlib import Path

# 專案根目錄 (本檔案所在處)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "ETH"
MODEL_DIR = ROOT_DIR / "data" / "models"

# 確保資料夾存在
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 交易對與時間框
SYMBOL = "ETH/USDT"
TIMEFRAME = "1d"  # Binance 單日線
START_DATE = "2014-01-01"  # 抓取起始日期
