import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Загрузка данных
df = pd.read_excel("../data/cs2_market_cap_analysis.xlsx", sheet_name="Исторические данные", skiprows=3)
df['Дата'] = pd.to_datetime(df['Дата'])
df = df.sort_values('Дата').set_index('Дата')

# Выделяем целевой ряд (в миллиардах для читаемости графика)
series = df['Капитализация ($)'] / 1e9

# 2. Обучение модели Хольта-Винтерса
# trend='add' — учитываем линейный тренд
# seasonal='add', seasonal_periods=30 — закладываем месячный цикл колебаний рынка
model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=30)
fitted_model = model.fit()

# 3. Прогноз на 365 дней вперед
forecast_days = 90
forecast_dates = pd.date_range(start=series.index.max() + pd.Timedelta(days=1), periods=forecast_days)
forecast_values = fitted_model.forecast(steps=forecast_days)

# 4. Визуализация
plt.figure(figsize=(14, 7))
plt.plot(series.index, series.values, label='Исторические данные', color='#10b981', linewidth=2)
plt.plot(forecast_dates, forecast_values, label='Реалистичный прогноз (Holt-Winters)', color='#3b82f6', linestyle='--', linewidth=2)

plt.title('Реалистичный прогноз капитализации CS2 (Модель Хольта-Винтерса)', fontsize=14, pad=15)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Капитализация (Млрд $)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
