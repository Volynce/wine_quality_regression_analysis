import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
red_wine_data = pd.read_csv('../data/winequality-red.csv', delimiter=';')
white_wine_data = pd.read_csv('../data/winequality-white.csv', delimiter=';')

# Объединение данных (по желанию можно работать с каждым датасетом отдельно)
wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# Проверка на пропущенные значения
print("Пропущенные значения в данных:\n", wine_data.isnull().sum())

# Обработка пропущенных значений (если есть)
# Можно заменить пропущенные значения на медиану каждого столбца
wine_data = wine_data.fillna(wine_data.median())

# Разделение данных на признаки (X) и целевую переменную (y)
X = wine_data.drop('quality', axis=1)  # Признаки, исключая колонку 'quality'
y = wine_data['quality']  # Целевая переменная (качество вина)

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Сохранение подготовленных данных (если нужно)
# Например, сохраняем X_train и y_train в CSV-файлы
# pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
# pd.DataFrame(y_train).to_csv('y_train.csv', index=False)

print("Данные успешно загружены, очищены и подготовлены.")
