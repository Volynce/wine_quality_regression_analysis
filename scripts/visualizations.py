import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных (предполагаем, что данные уже загружены)
red_wine_data = pd.read_csv('../data/winequality-red.csv', delimiter=';')
white_wine_data = pd.read_csv('../data/winequality-white.csv', delimiter=';')

# Объединение данных
wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# Визуализация распределения качества вина
def plot_quality_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['quality'], kde=True, color='blue')
    plt.title('Распределение качества вина')
    plt.xlabel('Качество')
    plt.ylabel('Частота')
    plt.savefig('wine_quality_distribution.png')  # Сохраняем график
    plt.show()

# Корреляционная матрица между признаками
def plot_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Корреляционная матрица')
    plt.savefig('correlation_matrix.png')  # Сохраняем график
    plt.show()

# Визуализация боксплота для анализа алкоголя по качеству
def plot_alcohol_by_quality(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='quality', y='alcohol', data=data)
    plt.title('Содержание алкоголя по качеству вина')
    plt.xlabel('Качество')
    plt.ylabel('Алкоголь')
    plt.savefig('alcohol_quality_boxplot.png')  # Сохраняем график
    plt.show()

# Гистограмма для фиксированного кислотного содержания
def plot_fixed_acidity_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['fixed acidity'], kde=True, color='green')
    plt.title('Распределение фиксированного кислотного содержания')
    plt.xlabel('Фиксированное кислотное содержание')
    plt.ylabel('Частота')
    plt.savefig('fixed_acidity_distribution.png')  # Сохраняем график
    plt.show()

# Влияние цитрусовой кислоты на качество вина
def plot_citric_acid_quality_scatter(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='citric acid', y='quality', data=data, color='purple')
    plt.title('Влияние цитрусовой кислоты на качество вина')
    plt.xlabel('Цитрусовая кислота')
    plt.ylabel('Качество')
    plt.savefig('citric_acid_quality_scatterplot.png')  # Сохраняем график
    plt.show()

# Функция для генерации всех графиков
def generate_all_visualizations(data):
    plot_quality_distribution(data)
    plot_correlation_matrix(data)
    plot_alcohol_by_quality(data)
    plot_fixed_acidity_distribution(data)
    plot_citric_acid_quality_scatter(data)

# Вызов функции для генерации всех графиков
generate_all_visualizations(wine_data)
