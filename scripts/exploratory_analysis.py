import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных (предполагаем, что данные уже загружены и подготовлены)
red_wine_data = pd.read_csv('../data/winequality-red.csv', delimiter=';')
white_wine_data = pd.read_csv('../data/winequality-white.csv', delimiter=';')

# Объединение данных (по желанию можно работать с каждым датасетом отдельно)
wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# Анализ распределения качества вина
plt.figure(figsize=(10, 6))
sns.histplot(wine_data['quality'], kde=True, color='blue')
plt.title('Распределение качества вина')
plt.xlabel('Качество')
plt.ylabel('Частота')
plt.savefig('wine_quality_distribution.png')  # Сохраняем график
plt.show()

# Корреляция между признаками и качеством вина
correlation_matrix = wine_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица')
plt.savefig('correlation_matrix.png')  # Сохраняем график
plt.show()

# Boxplot для анализа содержания алкоголя по качеству
plt.figure(figsize=(10, 6))
sns.boxplot(x='quality', y='alcohol', data=wine_data)
plt.title('Содержание алкоголя по качеству вина')
plt.xlabel('Качество')
plt.ylabel('Алкоголь')
plt.savefig('alcohol_quality_boxplot.png')  # Сохраняем график
plt.show()

# Распределение уровня фиксированного кислотного содержания
plt.figure(figsize=(10, 6))
sns.histplot(wine_data['fixed acidity'], kde=True, color='green')
plt.title('Распределение фиксированного кислотного содержания')
plt.xlabel('Фиксированное кислотное содержание')
plt.ylabel('Частота')
plt.savefig('fixed_acidity_distribution.png')  # Сохраняем график
plt.show()

# График распределения для других характеристик
plt.figure(figsize=(10, 6))
sns.scatterplot(x='citric acid', y='quality', data=wine_data, color='purple')
plt.title('Влияние цитрусовой кислоты на качество вина')
plt.xlabel('Цитрусовая кислота')
plt.ylabel('Качество')
plt.savefig('citric_acid_quality_scatterplot.png')  # Сохраняем график
plt.show()
