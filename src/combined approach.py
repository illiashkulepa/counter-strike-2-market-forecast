import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("../data/cs2_market_cap_analysis.xlsx", sheet_name="Исторические данные", skiprows=3)
df['Дата'] = pd.to_datetime(df['Дата'])
df = df.sort_values('Дата').reset_index(drop=True)

# Считаем средний исторический прирост в день (исключая аномальный шум)
df['Прирост'] = df['Капитализация ($)'].diff()
avg_daily_growth = df['Прирост'].median() # используем медиану, чтобы октябрьский краш не ломал логику

# Стартовая точка — последнее сглаженное значение из таблицы
start_val = df['30-дн. Скользящее среднее ($)'].iloc[-1] / 1e9
last_date = df['Дата'].max()

# Прогноз на 365 дней
forecast_days = 90
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]

# Строим прогнозную прямую: Старт + День * Средний_прирост
future_vals = [start_val + (i * avg_daily_growth / 1e9) for i in range(1, forecast_days + 1)]

# Визуализация
plt.figure(figsize=(14, 7))
plt.plot(df['Дата'], df['Капитализация ($)'] / 1e9, label='Исторические данные', color='#10b981')
plt.plot(future_dates, future_vals, label='Консервативный линейный прогноз', color='#f59e0b', linestyle='--', linewidth=2)

plt.title('Консервативный прогноз развития рынка скинов CS2', fontsize=14, pad=15)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Капитализация (Млрд $)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.show()