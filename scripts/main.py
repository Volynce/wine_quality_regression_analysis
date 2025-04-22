import os
import subprocess

def run_script(script_name):
    """
    Запускает указанный Python скрипт.
    """
    print(f"Запуск скрипта: {script_name}")
    subprocess.run(['python', script_name], check=True)

def main():
    # Путь к каждому из скриптов
    scripts = [
        'data_preprocessing.py',      # Скрипт для предобработки данных
        'exploratory_analysis.py',    # Скрипт для разведочного анализа
        'regression_model.py',        # Скрипт для обучения модели
        'visualizations.py'           # Скрипт для визуализаций
    ]
    
    # Запуск всех скриптов по очереди
    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    main()
