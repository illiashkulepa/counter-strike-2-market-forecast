import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# 1. Загрузка данных (Prophet требует строго колонки 'ds' для дат и 'y' для значений)
df = pd.read_excel("../data/cs2_market_cap_analysis.xlsx", sheet_name="Исторические данные", skiprows=3)
df_prophet = pd.DataFrame()
df_prophet['ds'] = pd.to_datetime(df['Дата'])
df_prophet['y'] = df['Капитализация ($)'] / 1e9

# 2. Инициализация и обучение модели
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(df_prophet)

# 3. Создание датафрейма для прогноза на год (365 дней)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# 4. Построение графика штатными средствами Prophet + кастомизация
fig, ax = plt.subplots(figsize=(14, 7))
model.plot(forecast, ax=ax)

# Перекрасим линии для наглядности
ax.get_lines()[0].set_color('#10b981') # Точки истории
ax.get_lines()[1].set_color('#2563eb') # Линия прогноза

plt.title('Прогноз рынка CS2 с помощью алгоритма Prophet (Meta)', fontsize=14, pad=15)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Капитализация (Млрд $)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()