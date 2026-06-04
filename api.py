from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import CORS
import numpy as np
import torch
from pathlib import Path
app = Flask(__name__)
CORS(app)
from ai import load_artifacts, forecast_prices  # импортируем из ai.py

app = Flask(__name__)
CORS(app)  # разрешаем запросы отовсюду

# Загружаем модель и всё, что нужно (один раз при старте)
MODEL_PATH = Path("artifacts")
model, scaler, config = load_artifacts(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

@app.route('/predict', methods=['GET'])
def predict():
    """Пример запроса: /predict?prices=100,101,102&days=14"""
    # Получаем параметры
    prices_str = request.args.get('prices')
    days = request.args.get('days', default=14, type=int)
    
    if not prices_str:
        return jsonify({"error": "Не передан параметр prices"}), 400
    
    # Преобразуем "100,101,102" в список float
    try:
        prices_list = [float(x.strip()) for x in prices_str.split(',')]
    except:
        return jsonify({"error": "Цены должны быть числами через запятую"}), 400
    
    if len(prices_list) < config.lookback:
        return jsonify({
            "error": f"Недостаточно данных. Нужно минимум {config.lookback} цен"
        }), 400
    
    # Превращаем в numpy массив для forecast_prices
    prices_array = np.array(prices_list).reshape(-1, 1)
    
    # Прогноз на days дней вперёд
    future = forecast_prices(
        model, scaler, prices_array,
        lookback=config.lookback,
        days=days,
        device=device
    )
    
    # Ответ
    return jsonify({
        "prediction": future.tolist(),
        "days": days,
        "last_price": prices_list[-1],
        "message": f"Прогноз на {days} дней"
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "artifacts_loaded": True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)