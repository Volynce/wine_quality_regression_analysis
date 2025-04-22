import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Загрузка данных (предполагаем, что данные уже загружены и подготовлены)
red_wine_data = pd.read_csv('../data/winequality-red.csv', delimiter=';')
white_wine_data = pd.read_csv('../data/winequality-white.csv', delimiter=';')

# Объединение данных (по желанию можно работать с каждым датасетом отдельно)
wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# Разделение данных на признаки (X) и целевую переменную (y)
X = wine_data.drop('quality', axis=1)  # Признаки, исключая колонку 'quality'
y = wine_data['quality']  # Целевая переменная (качество вина)

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Инициализация модели линейной регрессии
model = LinearRegression()

# Обучение модели
model.fit(X_train, y_train)

# Прогнозирование на тестовых данных
y_pred = model.predict(X_test)

# Оценка производительности модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse}")
print(f"Коэффициент детерминации (R²): {r2}")

# Для визуализации результатов (например, фактические vs предсказанные значения)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Фактические vs Предсказанные значения')
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.savefig('actual_vs_predicted.png')  # Сохраняем график
plt.show()
