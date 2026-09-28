# Avito AntiBot — Colab project

Структура следует формату референса krazy-train/Avito_bot_detection: основной notebook в корне, рядом модули признаков, метрики, pipeline, проверка submission, requirements и README. Код референса не копировался.

## Структура проекта

.
├── main_pipeline.ipynb
├── features.py
├── metric.py
├── pipeline.py
├── submission.py
├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── events_part1.csv.gz
│   ├── events_part2.csv.gz
│   └── events_part3.csv.gz
└── outputs/                 создаётся при запуске
    ├── submission.csv
    └── validation/

## Google Colab

Откройте main_pipeline.ipynb и запустите ячейки сверху вниз. Если runtime не содержит проектные файлы, notebook попросит загрузить Avito_AntiBot_Colab.zip. Пути менять не нужно. После завершения submission.csv скачивается автоматически.

## Локальный запуск

    python -m pip install -r requirements.txt
    jupyter notebook main_pipeline.ipynb

## ML-логика

Сохранены генераторы признаков, две конфигурации LightGBM, 10 seed на каждую модель, temporal validation с holdout от 2026-04-17, Precision@Recall>=0.70 и равновесный rank blend. Ожидаемый validation score: 0.8484848484848485.

Отдельный threshold не подбирается: официальный код считает метрику по всем допустимым score-порогам, а submission содержит непрерывный score.

## Файлы

| Файл | Назначение | Нужен для pipeline |
|---|---|---:|
| data/train.csv, data/test.csv, data/events_part*.csv.gz | Исходные данные | Да |
| features.py, metric.py, pipeline.py, submission.py | Код решения | Да |
| outputs/validation/data_audit.json | Аудит окна, дублей и типов событий | Нет, диагностический отчёт |
| outputs/validation/features_*.json | Список колонок матриц | Нет, диагностический отчёт |
| outputs/validation/robustness*.csv | Метрики seed и ансамбля | Нет, диагностический отчёт |
| outputs/validation/validation_metrics.json | Итоговые метрики holdout | Нет, диагностический отчёт |
| outputs/validation/validation_predictions.csv | Прогнозы holdout для анализа ошибок | Нет, диагностический отчёт |
| outputs/submission.csv | Итог соревнования | Результат |

Сохранение 20 текстовых моделей и их manifest исключено: текущий pipeline их не загружает для формирования submission. Кэш матриц признаков не создаётся.
