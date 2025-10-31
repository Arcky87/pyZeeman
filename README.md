# pyZeeman

Пайплайн для обработки двумерных астрономических спектров, полученных с образ-срезателя и анализатора круговой поляризации (ОЗСП).

## Описание

pyZeeman предназначен для полной обработки спектров с ОЗСП от сырых 2D изображений до откалиброванных одномерных спектров с разделением на два ортогональных поляризованных луча.

## Возможности

- Автоматическая сортировка кадров по типу (BIAS, FLAT, ThAr, объекты)
- Удаление космических частиц (cosmic ray removal)
- Удаление рассеянного света (scatter light removal)
- Создание мастер-калибровок (bias, flat)
- Трассировка 14 спектральных порядков с алгоритмом GETXWD
- Автоматическое сопоставление ThAr с научными кадрами
- Интерактивная + автоматическая калибровка по длинам волн (ThAr)
- Применение 2D дисперсионного решения
- Объединение порядков в два вектора для ортогональных поляризаций

## Архитектура пайплайна

```
RAW FITS files
    ↓
1. lister.py               → Сортировка по типам
    ↓
2. trimmer.py              → Обрезка области интереса
    ↓
3. list_astroscrappy.py    → Удаление космических частиц
    ↓
4. medianer.py             → Мастер-BIAS и мастер-FLAT
    ↓
5. list_subtractor.py      → Вычитание калибровок
    ↓
6. backlong_zee.py         → Удаление рассеянного света (опционально)
    ↓
7. not_so_simple_tracer.py → Трассировка 14 порядков (GETXWD)
    ↓
8. match_thar_to_science.py → Сопоставление ThAr с кадрами
    ↓
9. thar_auto_calibration.py → Дисперсионное решение (авто)
   thar_calibration.py      → Дисперсионное решение (интерактивно, первый раз)
    ↓
10. apply_calibration.py    → Применение калибровки
    ↓
11. combine_orders.py       → Объединение в два вектора
    ↓
РЕЗУЛЬТАТ: file_1.fits, file_2.fits (ортогональные поляризации)
```

## Установка

### Требования

- Python 3.8+
- Git (для клонирования репозитория)

### Зависимости

```bash
pip install numpy scipy matplotlib astropy
pip install astroscrappy specutils spectres
```

Библиотеки:
- `numpy` - численные вычисления
- `scipy` - научные вычисления и оптимизация
- `matplotlib` - визуализация
- `astropy` - FITS файлы и астрономические вычисления
- `astroscrappy` - удаление космических частиц (L.A.Cosmic алгоритм)
- `specutils` - спектроскопические утилиты
- `spectres` - передискретизация спектров

## Быстрый старт

### 1. Подготовка данных

Поместите ваши FITS файлы в директорию с данными:

```
/data/Observations/my_observation/
    └── (ваши FITS файлы здесь)
```

### 2. Настройка конфигурации

Скопируйте пример конфигурации и отредактируйте под ваши нужды:

```bash
cp config_pyzeeman.json my_config.json
```

Основные параметры:
- `data_dir`: Путь к директории с данными
- `gain`: Коэффициент усиления камеры (e-/ADU)
- `ron`: Шум считывания (e-)
- `n_orders`: Количество порядков (14 для ОЗСП)
- `trim_area`: Область интереса [x1, x2, y1, y2]
- `reference_order`: Опорный порядок для калибровки (обычно 11)
- `atlas_file`: Путь к атласу линий ThAr

### 3. Запуск автоматического пайплайна

Полная обработка:

```bash
python reduce4me.py my_config.json
```

Это выполнит все этапы от препроцессинга до экстракции.

**Опции:**

```bash
# Пропустить определенные этапы
python reduce4me.py my_config.json --skip-preprocessing
python reduce4me.py my_config.json --skip-tracing
python reduce4me.py my_config.json --skip-calibration

# Выполнить только определенный этап
python reduce4me.py my_config.json --only-calibration
python reduce4me.py my_config.json --only-extraction
```

### 4. Интерактивная калибровка (первый ThAr)

При первом запуске пайплайн остановится на этапе калибровки и запросит интерактивное создание опорного решения (reference solution).

Следуйте инструкциям в графическом окне:
1. Найдите и идентифицируйте спектральные линии
2. Сопоставьте пики с линиями атласа ThAr
3. Постройте дисперсионное решение
4. Проверьте остатки и сохраните результат

Опорное решение сохранится в `CALIBRATIONS/reference_solution.json` и будет использоваться для автоматической калибровки всех последующих ThAr.

### 5. Результаты

После успешного завершения пайплайна в директории с данными будут созданы:

```
my_observation/
├── RAW/                  # Исходные файлы (если были перемещены)
├── temp/                 # Временные списки и обработанные кадры
├── TRACED_ORDERS/        # JSON файлы с трассировкой
├── CALIBRATIONS/         # Калибровочные решения
│   ├── reference_solution.json
│   └── o*_FULL.json
├── CALIBRATED/           # Откалиброванные спектры по порядкам
│   └── o*/order_*.fits
├── FINAL/                # Финальные векторы поляриметрии
│   ├── o*_1.fits        # Порядки 1-7 (первая поляризация)
│   └── o*_2.fits        # Порядки 8-14 (вторая поляризация)
└── pyzeeman_pipeline.log # Лог обработки
```

## Структура проекта

### Основные модули пайплайна

```
pyZeeman/
├── reduce4me.py              # Главный скрипт пайплайна
│
├── # Препроцессинг
├── lister.py                 # Сортировка кадров
├── trimmer.py                # Обрезка ROI
├── list_astroscrappy.py      # Удаление космических частиц
├── medianer.py               # Создание мастер-калибровок
├── list_subtractor.py        # Вычитание калибровок
├── backlong_zee.py           # Удаление рассеянного света
│
├── # Трассировка и калибровка
├── not_so_simple_tracer.py   # Трассировка порядков (GETXWD)
├── match_thar_to_science.py  # Сопоставление ThAr
├── thar_auto_calibration.py  # Автоматическая калибровка
├── thar_calibration.py       # Интерактивная калибровка
│
├── # Экстракция и объединение
├── apply_calibration.py      # Применение решения
├── combine_orders.py         # Объединение порядков
├── extract_order_spectrum.py # Экстракция отдельных порядков
│
├── # Вспомогательные модули
├── loaders.py                # Утилиты загрузки
├── visualize_trace.py        # Визуализация
│
├── # Утилиты и инструменты
├── create_calibration_solution.py # Ручное создание калибровок
├── plot_references.py        # Визуализация опорных срезов
├── diagnostics.py            # Диагностика данных
├── bias_hist.py              # Анализ bias
├── plot_diff_across.py       # Отладка различий
│
├── # Конфигурация и данные
├── config_example.json       # Пример конфигурации
├── config_pyzeeman.json      # Рабочая конфигурация
├── thar.dat                  # Атлас линий ThAr
│
└── # Документация
    ├── README.md             # Этот файл
    └── PROJECT_ANALYSIS.md   # Детальный анализ проекта
```

## Детальное описание модулей

### Препроцессинг

#### lister.py
Автоматически определяет тип кадров по ключевым словам FITS заголовка. Создает списки:
- `bias_list.txt` - кадры смещения
- `flat_list.txt` - плоские поля
- `thar_list.txt` - калибровочные лампы ThAr
- `obj_list.txt` - научные объекты

#### list_astroscrappy.py
Удаление космических частиц с помощью алгоритма L.A.Cosmic (библиотека astroscrappy).
Создаёт маски и очищенные изображения.

#### backlong_zee.py
Удаление рассеянного света методом линейной аппроксимации фона по краям детектора с последующей медианной фильтрацией.

### Трассировка

#### not_so_simple_tracer.py
Реализует алгоритм GETXWD (из пакета REDUCE) для трассировки спектральных порядков:
- Определение границ порядков по флэт-кадру
- Гауссова аппроксимация профилей
- Аппроксимация трасс полиномами Чебышева
- Сохранение в JSON формат

### Калибровка

#### match_thar_to_science.py
Автоматическое сопоставление кадров ThAr с научными кадрами по времени экспозиции. Добавляет ключевое слово `THAR_REF` в заголовки FITS.

#### thar_calibration.py
Интерактивный режим для построения дисперсионного решения:
- Поиск и фитинг линий гауссианами
- Интерактивная идентификация линий атласа
- Автоматическое добавление точек с разницей < 0.002 Å
- Построение полиномиальной модели
- Анализ остатков
- Сохранение WCS параметров

#### thar_auto_calibration.py
Автоматическая калибровка всех ThAr на основе опорного решения:
- Использует cross-correlation для поиска сдвига
- Автоматически идентифицирует линии
- Строит дисперсионное решение для всех 14 порядков
- Оценивает качество решения (RMS)

### Экстракция

#### apply_calibration.py
Применение калибровки:
- Загрузка 2D дисперсионного решения
- Flux-conserving передискретизация
- Расчет ошибок по формуле ПЗС
- Сохранение в FITS с WCS

#### combine_orders.py
Объединение порядков в векторы поляриметрии:
- Взвешенное усреднение по потоку в областях перекрытия
- Разделение на группы (1-7 и 8-14)
- Сохранение финальных векторов

## Параметры конфигурации

Пример файла `config_example.json`:

```json
{
  "data_dir": "/data/Observations/test_pyzeeman_final/",
  "device": "zeeman",
  "n_orders": 14,
  "gain": 2.78,
  "ron": 5.6,
  "s_bias_name": "s_bias.fits",
  "s_flat_name": "s_flat.fits",
  "atlas_file": "thar.dat",
  "reference_order": 4,
  "poly_degree": 5,
  "upper_orders": [1, 2, 3, 4, 5, 6, 7],
  "lower_orders": [8, 9, 10, 11, 12, 13, 14],
  "preprocessing": {
    "trim_area": [10, 615, 10, 4590],
    "flip": false,
    "cosmic_ray_removal": true,
	"cosmic_ray_params": {
    "sigclip": 4.5,
    "sigfrac": 0.3,
    "objlim": 5.0,
    "niter": 4
	},
 	"scatter_light": {
	   "enabled": true,
	   "border_width": 70,
	   "save_background": false
  		}
  },
  "tracing": {
	"n_points_for_fit": 10,
	"smooth": true,
	"smooth_sigma": 1.0,
	"getxwd_gauss": false,
	"overwrite": true,
	"save_format": "json",
	"visualization": {
       "show_point_fits": false,    
       "show_final_traces": false,  
       "save_plots": true           
        }
  },
  "calibration": {
    "peak_detection": {
      "prominence_sigma": 12.0,
      "width_range": [1.1, 4.5],
      "distance_pixels": 10
    },
    "matching": {
      "wavelength_tolerance": 0.48,
      "max_shift_pixels": 4.0
    },
    "quality_thresholds": {
      "rms_excellent": 0.05,
      "rms_good": 0.09,
      "rms_warning": 0.45
    },
    "visualization": {
      "save_plots": true,
      "output_format": "pdf",
      "dpi": 300
       }
    }
}
```

### Описание ключевых параметров

**Основные:**
- `data_dir` - директория с данными
- `gain` - усиление камеры (e-/ADU)
- `ron` - шум считывания (e-)
- `n_orders` - количество порядков
- `reference_order` - опорный порядок для калибровки
- `atlas_file` - путь к атласу линий ThAr
- `poly_degree` - степень полинома для дисперсионного решения

**Препроцессинг:**
- `trim_area` - область обрезки [x1, x2, y1, y2]
- `cosmic_ray_params` - параметры L.A.Cosmic
- `scatter_light` - параметры удаления рассеянного света

**Трассировка:**
- `n_points_for_fit` - точки для аппроксимации трасс
- `getxwd_gauss` - использовать гауссиану при GETXWD

**Калибровка:**
- `peak_detection` - параметры поиска линий
- `matching` - параметры сопоставления с атласом
- `quality_thresholds` - пороги качества RMS

## Частые проблемы и решения

### Проблема: "No module named 'astroscrappy'"

```bash
pip install astroscrappy
```

### Проблема: Не найдены спектральные порядки

Проверьте параметры трассировки в конфигурации:
- Увеличьте `smooth_sigma`
- Проверьте `n_orders`
- Проверьте качество флэт-кадра

### Проблема: Плохая калибровка (высокий RMS)

- Используйте больше опорных линий (минимум 10-15)
- Проверьте покрытие по всему диапазону длин волн
- Попробуйте другую степень полинома (3-5)
- Проверьте качество ThAr (достаточный S/N)

### Проблема: Интерактивная калибровка не запускается

Убедитесь, что:
- У вас есть X11 forwarding (для удаленного подключения)
- matplotlib backend поддерживает интерактивность
- Достаточно найденных пиков в спектре ThAr

### Проблема: Ошибка при объединении порядков

- Проверьте, что все порядки успешно откалиброваны
- Проверьте конфигурацию `upper_orders` и `lower_orders`
- Проверьте наличие перекрытий между порядками

## Утилиты

### create_calibration_solution.py
Создание калибровочных решений вручную для сложных случаев.

### plot_references.py
Визуализация всех опорных срезов для проверки качества калибровки.

### diagnostics.py
Диагностика качества данных, проверка дисторсий.

### bias_hist.py
Анализ распределения значений в bias кадрах.

## Примеры использования

### Полная обработка

```bash
# Создать конфигурацию
cp config_example.json my_config.json
# Отредактировать my_config.json

# Запустить пайплайн
python reduce4me.py my_config.json
```

### Повторная калибровка

```bash
# Пропустить препроцессинг и трассировку, только калибровка
python reduce4me.py my_config.json --only-calibration
```

### Только экстракция

```bash
# Если калибровка уже есть, только извлечь спектры
python reduce4me.py my_config.json --only-extraction
```
## Ссылки

- [REDUCE package](http://www.astro.uu.se/~piskunov/RESEARCH/REDUCE/) - алгоритм GETXWD
- [astroscrappy](https://github.com/astropy/astroscrappy) - удаление космических частиц
- [specutils](https://specutils.readthedocs.io/) - спектроскопические утилиты
- [spectres](https://github.com/ACCarnall/SpectRes) - передискретизация спектров
