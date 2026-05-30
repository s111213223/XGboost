"""
=========================================================================
血糖風險預測 API (給 Flutter App 呼叫)
=========================================================================
特徵設計:
  - 5 營養素: Calories, Carbs, Protein, Fat, Fiber (使用者輸入)
  - 1 時間:   Hour (0-23, App 自動取得)

如何啟動:
  pip install fastapi uvicorn xgboost shap joblib pandas
  uvicorn api:app --host 0.0.0.0 --port 8000

訓練模型範例 (在 Colab 跑完後存檔):
  import joblib
  joblib.dump({
      'model': model,
      'features': ['Calories','Carbs','Protein','Fat','Fiber','Hour'],
  }, 'model.pkl')
=========================================================================
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import xgboost 
import shap
import joblib

app = FastAPI(title="Blood Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================================
# 載入模型
# =========================================================================
bundle = joblib.load('model.pkl')
model = bundle['model']
FEATURES = bundle['features']  # ['Calories','Carbs','Protein','Fat','Fiber','Hour']
explainer = shap.TreeExplainer(model)
print(f"✅ 模型已載入,特徵: {FEATURES}")


# =========================================================================
# Flutter 送過來的資料
# =========================================================================
class MealInput(BaseModel):
    timestamp: str  # "2025-11-08T12:30:00" (App 自動帶,後端從中抽 hour)
    calories: float
    carbs: float
    protein: float
    fat: float
    fiber: float


# =========================================================================
# 🌟 白話解釋規則庫
# 不講「風險提高」這種模型語氣,改講「為什麼」
# =========================================================================
def explain_feature(feature: str, value: float, shap_value: float, risk_level: int) -> dict:
    """
    把每個特徵的影響翻譯成白話
    
    回傳:
      {
        'title': '這餐碳水偏多',          # 一句話標題
        'detail': '碳水化合物 80g 是血糖飆升的主因',  # 詳細說明
        'emoji': '🍚',                    # 視覺提示
        'is_bad': True,                   # 是壞影響嗎
      }
    """
    is_bad = shap_value > 0  # SHAP 正值代表把預測往「更危險的類別」推

    # === Carbs (碳水) ===
    if feature == 'Carbs':
        if value > 70:
            return {
                'title': '這餐碳水偏多',
                'detail': f'你吃了 {value:.0f} 克碳水,超過一般建議的單餐 60 克,容易讓血糖快速上升',
                'emoji': '🍚',
                'is_bad': is_bad,
            }
        elif value > 40:
            return {
                'title': '碳水量適中',
                'detail': f'你吃了 {value:.0f} 克碳水,屬於正常範圍',
                'emoji': '🍚',
                'is_bad': is_bad,
            }
        else:
            return {
                'title': '碳水較少',
                'detail': f'你只吃了 {value:.0f} 克碳水,對血糖的衝擊較小',
                'emoji': '🍚',
                'is_bad': is_bad,
            }

    # === Fiber (纖維) ===
    if feature == 'Fiber':
        if value >= 8:
            return {
                'title': '纖維充足',
                'detail': f'你吃了 {value:.1f} 克纖維,有助於減緩血糖上升速度',
                'emoji': '🥬',
                'is_bad': is_bad,
            }
        elif value >= 3:
            return {
                'title': '纖維普通',
                'detail': f'這餐有 {value:.1f} 克纖維,可以再多吃點蔬菜',
                'emoji': '🥬',
                'is_bad': is_bad,
            }
        else:
            return {
                'title': '纖維不足',
                'detail': f'這餐只有 {value:.1f} 克纖維,缺乏蔬菜會讓血糖更容易飆升',
                'emoji': '🥬',
                'is_bad': is_bad,
            }

    # === Protein (蛋白質) ===
    if feature == 'Protein':
        if value >= 25:
            return {
                'title': '蛋白質充足',
                'detail': f'你吃了 {value:.0f} 克蛋白質,有助於穩定血糖',
                'emoji': '🥩',
                'is_bad': is_bad,
            }
        elif value >= 10:
            return {
                'title': '蛋白質普通',
                'detail': f'這餐有 {value:.0f} 克蛋白質',
                'emoji': '🥩',
                'is_bad': is_bad,
            }
        else:
            return {
                'title': '蛋白質偏少',
                'detail': f'這餐只有 {value:.0f} 克蛋白質,可以加點蛋或肉類',
                'emoji': '🥩',
                'is_bad': is_bad,
            }

    # === Fat (脂肪) ===
    if feature == 'Fat':
        if value > 30:
            return {
                'title': '脂肪偏多',
                'detail': f'你吃了 {value:.0f} 克脂肪,可能讓飽足感持久但熱量較高',
                'emoji': '🧈',
                'is_bad': is_bad,
            }
        else:
            return {
                'title': '脂肪量正常',
                'detail': f'這餐脂肪 {value:.0f} 克',
                'emoji': '🧈',
                'is_bad': is_bad,
            }

    # === Calories (熱量) ===
    if feature == 'Calories':
        if value > 700:
            return {
                'title': '這餐熱量偏高',
                'detail': f'總熱量 {value:.0f} 大卡,屬於大份量餐',
                'emoji': '🔥',
                'is_bad': is_bad,
            }
        elif value > 300:
            return {
                'title': '熱量適中',
                'detail': f'總熱量 {value:.0f} 大卡',
                'emoji': '🔥',
                'is_bad': is_bad,
            }
        else:
            return {
                'title': '熱量較少',
                'detail': f'總熱量只有 {value:.0f} 大卡,屬於輕食',
                'emoji': '🔥',
                'is_bad': is_bad,
            }

    # === Hour (用餐時間) ===
    if feature == 'Hour':
        hour = int(value)
        if 5 <= hour < 10:
            time_desc = '早餐時段'
        elif 10 <= hour < 14:
            time_desc = '午餐時段'
        elif 14 <= hour < 17:
            time_desc = '下午點心時段'
        elif 17 <= hour < 21:
            time_desc = '晚餐時段'
        else:
            time_desc = '宵夜時段'
        return {
            'title': f'用餐時間在{time_desc}',
            'detail': f'你在 {hour}:00 左右用餐,身體在不同時段對食物的反應不太一樣',
            'emoji': '⏰',
            'is_bad': is_bad,
        }

    return {
        'title': feature,
        'detail': f'{feature} = {value:.1f}',
        'emoji': '📊',
        'is_bad': is_bad,
    }


# =========================================================================
# 整體建議(根據預測結果給出生活化建議)
# =========================================================================
def get_overall_advice(risk_level: int, carbs: float, fiber: float) -> str:
    if risk_level == 0:
        return "這餐搭配得不錯!餐後可以放心活動,記得補充水分。"
    elif risk_level == 1:
        if fiber < 3:
            return "餐後建議散步 10-15 分鐘,下次可以多加點蔬菜幫助穩定血糖。"
        return "餐後建議散步 10-15 分鐘,讓身體慢慢消化。"
    else:  # risk_level == 2
        if carbs > 70:
            return "這餐碳水較多,建議餐後快走 20 分鐘以上,下一餐可以減少主食量。"
        return "建議餐後做點輕度運動,並注意接下來幾餐的飲食搭配。"


# =========================================================================
# 主要 API 端點
# =========================================================================
@app.post("/predict")
def predict(meal: MealInput):
    # 1. 從 timestamp 抽出 hour
    hour = pd.to_datetime(meal.timestamp).hour

    # 2. 組成模型輸入 (順序要和訓練時一致!)
    feature_values = {
        'Calories': meal.calories,
        'Carbs': meal.carbs,
        'Protein': meal.protein,
        'Fat': meal.fat,
        'Fiber': meal.fiber,
        'Hour': hour,
    }
    X = np.array([[feature_values[f] for f in FEATURES]])

    # 3. 預測
    pred_class = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    # 4. 算 SHAP
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_for_pred = shap_values[pred_class][0]
    else:
        shap_for_pred = shap_values[0, :, pred_class]

    # 5. 找出最重要的 3 個因素
    impacts = []
    for i, f in enumerate(FEATURES):
        impacts.append({
            'feature': f,
            'value': float(feature_values[f]),
            'shap': float(shap_for_pred[i]),
        })
    impacts.sort(key=lambda x: -abs(x['shap']))
    top_3 = impacts[:3]

    # 6. 翻譯成白話
    explanations = [
        explain_feature(f['feature'], f['value'], f['shap'], pred_class)
        for f in top_3
    ]

    # 7. 標題改成白話
    risk_titles = {
        0: '這餐很適合你',
        1: '這餐需要留意',
        2: '這餐建議調整',
    }
    risk_emoji = {0: '✅', 1: '⚠️', 2: '🚨'}

    return {
        'risk_level': pred_class,
        'risk_title': risk_titles[pred_class],
        'risk_emoji': risk_emoji[pred_class],
        'overall_advice': get_overall_advice(pred_class, meal.carbs, meal.fiber),
        'confidence': float(proba.max()),
        'reasons': explanations,
    }


@app.get("/")
def root():
    return {"status": "ok"}