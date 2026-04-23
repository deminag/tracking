# Maritime Object Tracking System

Веб-приложение для детекции и трекинга морских объектов на видео с использованием YOLO/RT-DETR и BYTETrack с фильтром Калмана.

## Возможности

- Загрузка видео через веб-интерфейс
- Выбор модели: YOLOv8x, YOLO26x, RT-DETR-X, RT-DETR-L
- Выбор девайса: CPU или CUDA
- Настройки трекера BYTETrack (track_thresh, match_thresh, track_buffer)
- Настройки фильтра Калмана (kf_q, kf_r, kf_p)
- Отображение прогресса обработки в реальном времени
- Подпись ID объекта и класса (для датасета aboships-plus)
- Треки движения объектов с визуализацией
- Метрики: количество треков, кадров, FPS, детекции, средняя уверенность, стабильность BB
- Сохранение результатов в PostgreSQL с историей сессий
- Просмотр истории обработок с метриками

## Структура проекта

```
app.py               # Основное приложение Flask, обработка видео, функции базы данных
tracker.py           # BYTETrack с фильтром Калмана
templates/index.html # Веб-интерфейс с настройками
weights/             # Обученные модели YOLO/RT-DETR
static/
  uploads/           # Загруженные видео файлы
  results/           # Обработанные видео с треками
trackeval/           # Библиотека TrackEval для оценки MOTChallenge
docker-compose.yml   # PostgreSQL 17 + Flask
Dockerfile           # Docker образ приложения
requirements.txt     # Python зависимости
training             # История обучения моделей
```

## Требования

- Python 3.10+
- PostgreSQL 17
- CUDA (опционально для GPU)
- Docker

## Установка и запуск

### Локальный запуск

1. Установите torch (требуется для CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# или для CPU
pip install torch torchvision
```

2. Установите остальные зависимости:
```bash
pip install -r requirements.txt
```
3. [Загрузите обученные модели и переместите в папку weights](https://drive.google.com/drive/folders/1__8Ykc9DKQ0WF4X8zYcI5geYlS0uaenT?usp=sharing) 

4. Создайте базу данных:
```bash
# Для локального запуска
docker run --name maritime_postgres -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=maritime_tracking \
  postgres:17

```

5. Запустите приложение:
```bash
python app.py
```

6. Откройте в браузере: `http://localhost:5000`

### Docker запуск

```bash
docker-compose up --build -d # Запуск в фоновом режиме
docker-compose logs -f       # Просмотр логов
docker-compose down          # Остановка

```
[Загрузите обученные модели и переместите в папку weights](https://drive.google.com/drive/folders/1__8Ykc9DKQ0WF4X8zYcI5geYlS0uaenT?usp=sharing) 

Приложение будет доступно: `http://localhost:5000`

## Архитектура

### Backend
- **Flask** - веб-фреймворк
- **YOLO/RT-DETR** - нейронные сети для детекции объектов
- **BYTETracker + Kalman Filter** - алгоритм трекинга с фильтром Калмана
- **PostgreSQL** - база данных для хранения результатов
- **OpenCV** - обработка видео и наложение треков

### Frontend
- **HTML/CSS/JavaScript** - адаптивный интерфейс
- **Real-time progress** - отображение прогресса обработки
- **Video playback** - встроенное воспроизведение результатов

## Использование

1. Выберите модель (YOLOv8x, YOLO26x, RT-DETR-X, RT-DETR-L)
2. Выберите девайс (CPU или CUDA)
3. Настройте параметры трекера и фильтра Калмана (опционально):
   - **BYTETracker параметры:**
     - track_thresh: порог уверенности детекции (по умолчанию 0.4)
     - match_thresh: порог IoU для сопоставления (по умолчанию 0.8)
     - track_buffer: буфер хранения потерянных треков (по умолчанию 190)
   - **Фильтр Калмана параметры:**
     - kf_q: шум процесса (по умолчанию 0.025)
     - kf_r: шум измерений (по умолчанию 0.1)
     - kf_p: начальная ковариация (по умолчанию 10.0)
4. Загрузите видео
5. После завершения результат отобразится на экране

## [Пример использования](https://drive.google.com/file/d/1Pa4HCZpCNLIbNHE-YOaSyRs65ynKdBmP/view?usp=sharing)

## Метрики

- **Количество уникальных треков** - число уникальных track_id за всё время обработки
- **Кадров** - общее количество кадров в видео
- **FPS (обработка)** - скорость обработки (кадров в секунду)
- **FPS (видео)** - частота кадров исходного видео
- **Количество срабатываний детектора** - общее число детекций на всех кадрах
- **Средняя уверенность** - средняя уверенность всех детекций (0-1)
- **Стабильность BB** - стабильность bounding box (IoU между кадрами, %)
- **История сессий** - просмотр всех предыдущих обработок с метриками
