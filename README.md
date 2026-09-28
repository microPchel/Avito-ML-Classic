# Avito AntiBot — ансамбль LightGBM + XGBoost

Решение задачи Avito Data Science Bootcamp по определению автоматизированного трафика по событиям `cookie_id` внутри 24-часового окна.

## Результат

На temporal validation:

```text
Precision @ Recall >= 0.70 = 0.848485
```

Финальный ансамбль объединяет три компонента:

| Компонент | Признаки | Вес |
|---|---:|---:|
| LightGBM SAFE | 202 | 40% |
| LightGBM SAFE + Pointer | 243 | 40% |
| XGBoost Top-150 | 150 | 20% |

Каждая модель обучается на 10 seeds:

```python
SEEDS = [0, 1, 7, 42, 123, 11, 29, 73, 2026, 3407]
```

Итого в финальном ансамбле используется 30 моделей. Их предсказания переводятся в ранги и объединяются weighted rank blend.

## Данные

Для запуска нужны:

```text
train.csv
test.csv
events.csv.gz
top150_train.csv.gz
top150_test.csv.gz
metric.py
```

В текущей версии notebook ожидает их в:

```text
/content/content/
```

Структура:

```text
/content/content/
├── train.csv
├── test.csv
├── events.csv.gz
├── top150_train.csv.gz
└── top150_test.csv.gz

/content/
└── metric.py
```

`top150_train.csv.gz` и `top150_test.csv.gz` содержат заранее подготовленные 150 признаков для XGBoost.

## Валидация

Используется временное разделение без случайного shuffle:

```text
TRAIN: window_start_ts < 2026-04-17
VALID: window_start_ts >= 2026-04-17
```

Train охватывает более ранний период, validation — последние дни train-выборки.

Основная метрика:

```text
Precision @ Recall >= 0.70
```

## Предобработка

Для каждой cookie используются только события из её собственного 24-часового окна:

```text
window_start_ts <= event_ts < window_end_ts
```

Также:

- исключается `captcha_shown`;
- удаляются только полные дубликаты строк;
- события с одинаковыми `(cookie_id, event_ts)` не удаляются;
- события сортируются по `cookie_id` и `event_ts`;
- target не используется при генерации признаков;
- не используются cross-cookie target statistics или многодневная история.

## Признаки

### SAFE — 202 признака

Основной набор поведенческих признаков:

- количество и доли типов событий;
- item/category/location diversity;
- поисковое поведение;
- временные интервалы между событиями;
- burstiness и регулярность;
- session-признаки;
- переходы между событиями;
- user-agent и platform;
- поведение внутри item;
- возраст cookie на начало окна.

### Pointer — 243 признака

К 202 SAFE-признакам добавляются 41 признак pointer-траектории:

- статистики `pointer_x` и `pointer_y`;
- радиус относительно центра;
- площадь и anisotropy;
- расстояния между позициями;
- скорость движения;
- stationary share;
- axis-aligned движения;
- straightness;
- cosine между последовательными направлениями.

Для одинаковых timestamp pointer-координаты агрегируются медианой.

### XGBoost Top-150

Отдельная XGBoost-модель обучается на заранее отобранной матрице из 150 признаков.

## Архитектура ансамбля

Текущие веса задаются в ячейке конфигурации:

```python
MODEL_WEIGHTS = {
    "safe": 0.40,
    "pointer": 0.40,
    "xgb": 0.20,
}
```

Для каждого компонента:

1. модель обучается на 10 seeds;
2. probability переводится в rank;
3. ranks суммируются внутри компонента;
4. три компонента объединяются с весами `40% / 40% / 20%`.

Validation score:

```text
SAFE seed ensemble:       0.818182
Pointer seed ensemble:    0.812950
XGBoost seed ensemble:    0.805755
Final rank blend:         0.848485
```

## Запуск

Открыть notebook:

```text
Avito_AntiBot_Best_084848_Colab_(5).ipynb
```

Загрузить необходимые файлы и выполнить:

```text
Runtime -> Run all
```

После обучения будет создан:

```text
/content/submission.csv
```

Формат:

```csv
cookie_id,score
ck_...,0.1234
ck_...,0.8765
```

В test находится 4909 cookies.

## Изменение весов

Веса меняются здесь:

```python
MODEL_WEIGHTS = {"safe": 0.40, "pointer": 0.40, "xgb": 0.20}
```

Например:

```python
MODEL_WEIGHTS = {"safe": 0.20, "pointer": 0.40, "xgb": 0.40}
```

После изменения нужно заново выполнить блок временной валидации и блок создания `submission.csv`.

Желательно, чтобы сумма весов была равна `1.0`.

## Важные ограничения

Решение специально избегает потенциальных утечек:

- нет событий после `window_end_ts`;
- нет `captcha_shown`;
- нет недельной истории cookie;
- нет item popularity по другим cookies;
- нет target encoding по `item_id`, category или query;
- нет cross-cookie статистик;
- нет использования test labels.

Все основные признаки описывают только поведение конкретной cookie в её собственном суточном окне.
