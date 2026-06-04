from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
from pathlib import Path
import traceback

from ai import load_artifacts, forecast_prices

app = Flask(__name__)
CORS(app)  # разрешаем запросы отовсюду

# Загружаем модель и всё, что нужно (один раз при старте)
MODEL_PATH = Path("artifacts")
model, scaler, config = load_artifacts(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

@app.route('/', methods=['GET'])
def home():
    """Корневой маршрут чтобы не было 404"""
    return jsonify({
        "status": "ok",
        "message": "NextPrice API работает. Используй /predict?prices=100,101,102&days=7"
    })

@app.route('/predict', methods=['GET'])
def predict():
    """Прогноз: /predict?prices=100,101,102&days=14"""
    try:
        prices_str = request.args.get('prices')
        days = request.args.get('days', default=14, type=int)

        if not prices_str:
            return jsonify({"error": "Не передан параметр prices"}), 400

        # Преобразуем строку в список float
        try:
            prices_list = [float(x.strip()) for x in prices_str.split(',')]
        except ValueError:
            return jsonify({"error": "Цены должны быть числами через запятую"}), 400

        if len(prices_list) < config.lookback:
            return jsonify({
                "error": f"Недостаточно данных. Нужно минимум {config.lookback} цен"
            }), 400

        # Вызываем твою функцию forecast_prices (правильный порядок аргументов)
        future = forecast_prices(
            prices_list,      # список цен
            days,             # количество дней
            model,            # модель LSTM
            scaler,           # scaler
            config,           # конфиг
            device            # устройство
        )

        # Преобразуем в список, если numpy/tensor
        if hasattr(future, 'tolist'):
            future = future.tolist()
        elif isinstance(future, np.ndarray):
            future = future.tolist()

        return jsonify({
            "prediction": future,
            "days": days,
            "last_price": prices_list[-1],
            "message": f"Прогноз на {days} дней"
        })

    except Exception as e:
        error_details = traceback.format_exc()
        return jsonify({
            "error": str(e),
            "trace": error_details
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "artifacts_loaded": True})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)