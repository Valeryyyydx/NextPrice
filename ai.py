"""
LSTM-модель для прогнозирования цен на PyTorch.

Модуль можно запускать напрямую или импортировать в приложение:

    from ai import PriceLSTM, train_model, forecast_prices, load_prices
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelConfig:
    lookback: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    test_size: float = 0.2
    patience: int = 8


class PriceLSTM(nn.Module):
    """Двухслойная LSTM для прогноза следующей цены."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_prices(
    ticker: str | None = "AAPL",
    csv_path: str | Path | None = None,
    period: str = "5y",
) -> tuple[np.ndarray, np.ndarray]:
    """Загружает цены закрытия из CSV или yfinance."""
    if csv_path:
        df = pd.read_csv(csv_path)
        if "Close" not in df.columns and "Price" in df.columns:
            df["Close"] = df["Price"]
        if "Date" not in df.columns:
            raise ValueError("CSV должен содержать колонку Date")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        prices = df["Close"].astype(float).values.reshape(-1, 1)
        dates = df["Date"].values
        return prices, dates

    if not ticker:
        raise ValueError("Укажите ticker или csv_path")

    import yfinance as yf

    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"Не удалось загрузить данные для {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    prices = df["Close"].astype(float).values.reshape(-1, 1)
    dates = df.index.values
    return prices, dates


def create_sequences(
    data: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Формирует обучающие окна: X = прошлые lookback дней, y = следующий день."""
    x, y = [], []
    for i in range(lookback, len(data)):
        x.append(data[i - lookback : i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


def _split_sequences(
    x: np.ndarray, y: np.ndarray, test_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(x) * (1 - test_size))
    return x[:split], x[split:], y[:split], y[split:]


def _to_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    prices: np.ndarray,
    config: ModelConfig | None = None,
    device: torch.device | None = None,
) -> tuple[PriceLSTM, MinMaxScaler, dict]:
    """Обучает LSTM и возвращает модель, scaler и метрики."""
    config = config or ModelConfig()
    device = device or get_device()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    x, y = create_sequences(scaled, config.lookback)
    x_train, x_test, y_train, y_test = _split_sequences(x, y, config.test_size)

    train_loader = _to_loader(x_train, y_train, config.batch_size, shuffle=True)
    test_loader = _to_loader(x_test, y_test, config.batch_size, shuffle=False)

    model = PriceLSTM(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _ in range(config.epochs):
        model.train()
        train_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x).squeeze(-1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x).squeeze(-1)
                val_losses.append(criterion(preds, batch_y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate_model(model, x_test, y_test, scaler, device)
    metrics["best_val_loss"] = best_val_loss
    metrics["history"] = history

    return model, scaler, metrics


@torch.no_grad()
def predict_next(
    model: PriceLSTM,
    window: np.ndarray,
    device: torch.device | None = None,
) -> float:
    """Предсказывает одно следующее значение по окну scaled-данных."""
    device = device or get_device()
    model.eval()
    x = torch.tensor(window, dtype=torch.float32).reshape(1, -1, 1).to(device)
    return float(model(x).item())


def forecast_prices(
    model: PriceLSTM,
    scaler: MinMaxScaler,
    prices: np.ndarray,
    lookback: int,
    days: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Авторегрессионный прогноз на несколько дней вперёд."""
    device = device or get_device()
    scaled = scaler.transform(prices)
    window = scaled[-lookback:].copy()
    predictions: list[float] = []

    for _ in range(days):
        next_scaled = predict_next(model, window, device)
        predictions.append(next_scaled)
        window = np.vstack([window[1:], [[next_scaled]]])

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


@torch.no_grad()
def evaluate_model(
    model: PriceLSTM,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or get_device()
    model.eval()

    x_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1).to(device)
    y_pred_scaled = model(x_tensor).cpu().numpy().flatten()

    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_artifacts(
    model: PriceLSTM,
    scaler: MinMaxScaler,
    config: ModelConfig,
    path: str | Path = "artifacts",
) -> None:
    """Сохраняет модель, scaler и конфиг для использования в приложении."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path / "lstm_model.pt")
    np.save(path / "scaler_min.npy", scaler.min_)
    np.save(path / "scaler_scale.npy", scaler.scale_)
    (path / "config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_artifacts(
    path: str | Path = "artifacts",
    device: torch.device | None = None,
) -> tuple[PriceLSTM, MinMaxScaler, ModelConfig]:
    path = Path(path)
    device = device or get_device()

    config = ModelConfig(**json.loads((path / "config.json").read_text(encoding="utf-8")))
    model = PriceLSTM(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(torch.load(path / "lstm_model.pt", map_location=device))
    model.to(device).eval()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.min_ = np.load(path / "scaler_min.npy")
    scaler.scale_ = np.load(path / "scaler_scale.npy")
    scaler.n_features_in_ = 1

    return model, scaler, config


#def plot_forecast(
 #   dates: np.ndarray,
#    prices: np.ndarray,
 #   lookback: int,
 #   test_size: float,
  #  y_true: Iterable[float] | None = None,
 #   y_pred: Iterable[float] | None = None,
  #  future_prices: Iterable[float] | None = None,
  #  output_dir: str | Path = ".",
#) -> None:
 #   output_dir = Path(output_dir)
 #   if y_true is not None and y_pred is not None:
  #      y_true_arr = np.array(list(y_true))
  #      y_pred_arr = np.array(list(y_pred))
  #      # просто берем последние даты по длине предсказаний
 ##       test_dates = dates[-len(y_true_arr):]
  #      plt.figure(figsize=(12, 5))
  #      plt.plot(test_dates, y_true_arr, label="Факт")
  #      plt.plot(test_dates, y_pred_arr, label="Прогноз LSTM")
   #     plt.title("Качество модели на тестовой выборке")
#        plt.xticks(rotation=45)
 #       plt.tight_layout()
   #     plt.savefig(output_dir / "test_forecast.png", dpi=150)
   #     plt.close()
   # if future_prices is not None:
   #     future = np.array(list(future_prices))
   #     recent = prices[-120:, 0] if prices.ndim > 1 else prices[-120:]
   #     plt.figure(figsize=(12, 5))
   #     plt.plot(range(len(recent)), recent, label="История")
   #     plt.plot(
   #         range(len(recent), len(recent) + len(future)),
   #         future,
   #         label="Прогноз",
   #         linestyle="--",
   #     )
   #     plt.title(f"Прогноз на {len(future)} дней вперёд")
   #     plt.xlabel("День")
   #     plt.ylabel("Цена")
   #     plt.legend()
   #     plt.tight_layout()
   #     plt.savefig(output_dir / "future_forecast.png", dpi=150)
   #     plt.close()
   # print("\nСохранено: artifacts/lstm_model.pt, test_forecast.png, future_forecast.png")
#
#
if __name__ == "__main__":
    main()
