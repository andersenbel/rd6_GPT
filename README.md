# Звіт по виконанню домашнього завдання

## Мета завдання

Метою цього завдання було:
1. Підготовка набору даних для fine-tuning GPT-2.
2. Налаштування та запуск процесу навчання.
3. Аналіз та оцінка результатів тренування.
4. Демонстрація результатів у вигляді звіту.

---

## Пункти домашнього завдання

### 1. **Підготовка та налаштування проекту**
#### Виконано:
- Встановлено всі необхідні бібліотеки (`torch`, `transformers`, `datasets`).
- Організовано структуру проекту

---

### Опис файлів
- **`dataset_prep.py`**:
  - Завантажує набір даних (IMDB).
  - Розділяє його на три частини: `train`, `val`, `test`.
  - Токенізує дані з використанням токенізатора GPT-2.
  - Форматує дані для використання у PyTorch.

- **`fine_tune.py`**:
  - Завантажує попередньо треновану модель GPT-2.
  - Налаштовує параметри навчання (`learning_rate`, `batch_size`, `epochs`).
  - Використовує `Trainer` для тренування моделі.

- **`train.py`**:
  - Інтегрує функції з `dataset_prep.py` та `fine_tune.py`.
  - Забезпечує запуск процесу тренування.
  - Логує проміжні результати та статус.

---

## Пункти виконання завдання

### 1. Підготовка та налаштування проекту
- Створено відповідну структуру директорій та файлів.
- Встановлено всі необхідні бібліотеки (`torch`, `transformers`, `datasets`).
- Налаштовано середовище для тренування моделі.

### 2. Підготовка даних
- Завантажено набір даних IMDB:
  - **Train**: 16,000 записів.
  - **Validation**: 2,000 записів.
  - **Test**: 2,000 записів.
- Дані токенізовано та відформатовано для PyTorch.

### 3. Тренування моделі
- Запущено процес навчання моделі GPT-2 з параметрами:
  - **learning_rate**: `5e-5`
  - **batch_size**: `4`
  - **epochs**: `3`

### 4. Аналіз та результати
- Успішно проведено тренування за 3 епохи.
- Основні проблеми:
  - Помилка з розмірами батчів (`Expected input batch_size to match target batch_size`).
  - Виправлено, додавши `pad_token` до токенізатора GPT-2.

---

## Результати

Після виконання скрипта `train.py`:

### 1. Логи навчання
Усі ключові метрики тренування, включно з втратою (loss) і точністю (accuracy), збережені у вигляді логів у директорії: `/logs/`

Логи містять детальну інформацію про навчання моделі на кожній епосі:
- Втрата для тренувального та валідаційного наборів даних.
- Час виконання кожної епохи.
- Значення гіперпараметрів, які використовувались (розмір батчу, кількість епох, швидкість навчання тощо).

### 2. Модель
Фінальна модель, навчена на основі вибраних даних, зберігається у директорії: `/results/model/`

Ця директорія містить:
- `pytorch_model.bin` — ваги моделі, які можна використати для прогнозування.
- `config.json` — конфігурація моделі, що описує її архітектуру.
- `tokenizer_config.json` — конфігурація токенізатора.
- `vocab.json` і `merges.txt` — файли, необхідні для роботи токенізатора.

### 3. Токенізовані дані
Підготовлені для тренування дані збережені у форматі `torch.Tensor` у директорії: `/data/tokenized/`

Це включає:
- Тренувальний набір (`train_dataset.pt`).
- Валідаційний набір (`val_dataset.pt`).
- Тестовий набір (`test_dataset.pt`).

Ці файли можна використовувати для повторного тренування або тестування моделі.

### 4. Графіки метрик
Для візуалізації прогресу навчання створено графіки втрат (loss) і точності (accuracy) для тренувального та валідаційного наборів. Ці графіки зберігаються у форматі `.png` у директорії:

Приклад згенерованих графіків:
- `loss_curve.png` — графік зміни втрати на кожній епосі.
- `accuracy_curve.png` — графік зміни точності моделі на валідаційних даних.

---

## Як використовувати отримані результати

1. **Тестування моделі**:
   - Завантажте модель з файлу `pytorch_model.bin` і застосуйте її для прогнозування на нових даних.
   - Використовуйте токенізатор, файли якого зберігаються у директорії `results/model/`.

2. **Оцінка точності**:
   - Використайте тестовий набір даних (`test_dataset.pt`) для оцінки якості передбачень моделі.

3. **Повторне тренування**:
   - Для швидкого повторного тренування використайте підготовлені токенізовані дані з директорії `/data/tokenized/`.

4. **Аналіз навчання**:
   - Перегляньте графіки у папці `/results/plots/`, щоб оцінити, чи досягла модель стабільності в навчанні, і визначити оптимальну кількість епох.

---

## Висновки

Робота скрипта дозволяє отримати повний набір результатів навчання:
- Фінальна модель готова до використання в реальних завданнях.
- Логи та графіки дозволяють провести аналіз ефективності тренування.
- Збережені токенізовані дані забезпечують можливість подальшого використання моделі або покращення її результатів.
