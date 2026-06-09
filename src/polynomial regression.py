import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. Загрузка данных из Excel
df = pd.read_excel("../data/cs2_market_cap_analysis.xlsx", sheet_name="Исторические данные", skiprows=3)

# Преобразуем дату в формат datetime и сортируем по возрастанию
df['Дата'] = pd.to_datetime(df['Дата'])
df = df.sort_values('Дата').reset_index(drop=True)

# 2. Подготовка данных для регрессии
# Компьютер не умеет строить регрессию напрямую по датам,
# поэтому переводим их в количество дней от начала графика (0, 1, 2...)
df['День'] = (df['Дата'] - df['Дата'].min()).dt.days

X = df['День'].values
y = df['Капитализация ($)'].values

# 3. Настройка степени полинома
# Степень 2 (квадратичная) покажет плавный глобальный тренд.
# Степени 3-5 будут сильнее изгибаться под конец графика. Сделайте 2 или 3.
degree = 2

# Находим коэффициенты полинома (от старшей степени к младшей)
coefficients = np.polyfit(X, y, degree)
# Создаем функцию полинома на основе найденных коэффициентов
poly_model = np.poly1d(coefficients)

# 4. Формирование сетки для прогноза на 365 дней вперед
forecast_days = 90
last_day = X[-1]
last_date = df['Дата'].max()

# Создаем массив дней от 0 до (последний день + 365)
X_future = np.arange(0, last_day + forecast_days + 1)
# Генерируем соответствующие даты для будущего периода
dates_future = [df['Дата'].min() + timedelta(days=int(d)) for d in X_future]

# Считаем значения капитализации по нашей модели для всей шкалы
y_future_pred = poly_model(X_future)

# 5. Выделение прогнозного хвоста в отдельный DataFrame (для анализа)
df_forecast = pd.DataFrame({
    'Дата': dates_future[len(X):],
    'Прогноз Капитализации ($)': y_future_pred[len(X):].astype(int)
})

print("--- ПЕРВЫЕ 5 ДНЕЙ ПРОГНОЗА ---")
print(df_forecast.head())

# 6. Визуализация результатов
plt.figure(figsize=(14, 7))

# Исторические реальные данные
plt.plot(df['Дата'], y / 1e9, label='Исторические данные (Excel)', color='#10b981', linewidth=2)

# Линия полиномиальной регрессии (прошлое + прогноз)
plt.plot(dates_future, y_future_pred / 1e9,
         label=f'Полиномиальная регрессия ({degree} степень) + Прогноз',
         color='#ef4444', linestyle='--', linewidth=2)

# Визуальная граница начала прогноза
plt.axvline(x=last_date, color='gray', linestyle=':', label='Начало прогноза (Май 2026)')

plt.title(f'Прогноз капитализации рынка CS2 методом полиномиальной регрессии ({degree}-й степени)', fontsize=14, pad=15)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Капитализация (Млрд $)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

# Показать график
plt.show()