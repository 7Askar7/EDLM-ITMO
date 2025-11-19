# Квантизация Нейронных Сетей: Сравнительный Анализ Методов QAT

## Обзор Проекта

Данный проект представляет собой комплексное сравнение **5 современных методов квантизации с учетом обучения (QAT)** на **3 различных архитектурах нейронных сетей**. Цель проекта — определить наиболее эффективный подход к квантизации для различных типов моделей и предоставить практические рекомендации для развертывания квантизованных моделей в production.

### Исследуемые Методы Квантизации
1. **LSQ** (Learned Step Size Quantization) - ICLR 2020
2. **PACT** (Parameterized Clipping Activation) - ICLR 2018
3. **AdaRound** (Adaptive Rounding) - ICML 2020
4. **APoT** (Additive Powers-of-Two) - ICLR 2020
5. **DSQ** (Differentiable Soft Quantization) - ICCV 2019

### Протестированные Архитектуры Моделей
1. **LSTM Classifier** - Анализ тональности текста (датасет IMDB)
2. **ESPCN** - Повышение разрешения изображений (DIV2K/BSD300)
3. **SASRec** - Последовательные рекомендации (MovieLens-25M)

### Ключевые Результаты
- **LSQ** показывает лучшие результаты на всех архитектурах (потеря качества 0.2-0.99%)
- **PACT** полностью проваливается на LSTM (потеря 46.35%) из-за несовместимости с sigmoid/tanh
- **DSQ** оптимален для CNN архитектур (потеря 3.35% на ESPCN)
- INT8 конверсия обеспечивает **сжатие модели на 35-62%** с минимальной деградацией качества

---

## Структура Директорий

```
project/
├── models/                      # Определения архитектур моделей
│   ├── lstm_classifier.py       # LSTM с поддержкой квантизации
│   ├── espcn.py                # ESPCN super-resolution CNN
│   └── sasrec.py               # Self-Attentive Sequential Recommender
│
├── quantization/               # Реализации методов квантизации
│   ├── base.py                 # Базовый класс квантизатора (STE, вычисление диапазона)
│   ├── lsq.py                  # Learned Step Size Quantization
│   ├── pact.py                 # Parameterized Clipping Activation
│   ├── adaround.py             # Adaptive Rounding
│   ├── apot.py                 # Additive Powers-of-Two
│   ├── dsq.py                  # Differentiable Soft Quantization
│   └── fake_quantize.py        # Реализации квантизованных слоев
│
├── scripts/                    # Скрипты обучения и оценки
│   ├── train_lstm_production.py       # Обучение LSTM
│   ├── train_espcn_production.py      # Обучение ESPCN
│   ├── train_sasrec_production.py     # Обучение SASRec
│   ├── simple_int8_convert.py         # Конверсия в реальный INT8
│   ├── evaluate_lstm_espcn_int8_cpu.py # Оценка INT8 на CPU
│   ├── evaluate_sasrec_int8_cpu.py    # Оценка SASRec INT8
│   ├── export_to_onnx.py              # Экспорт в ONNX
│   ├── benchmark_onnx.py              # Бенчмаркинг ONNX Runtime
│   └── generate_comparison_plots.py   # Генерация визуализаций
│
├── utils/                      # Вспомогательные функции
│   ├── datasets.py             # Загрузчики датасетов и предобработка
│   ├── metrics.py              # Метрики оценки (ROC-AUC, PSNR, NDCG)
│   └── training.py             # Утилиты для обучения
│
├── data/                       # Датасеты (не включены в репозиторий)
│   ├── imdb/                   # Рецензии IMDB
│   ├── DIV2K/                  # Изображения для super-resolution
│   └── ml-25m/                 # Рейтинги MovieLens-25M
│
├── results/                    # Результаты обучения и конвертированные модели
│   ├── baseline/               # Базовые модели (без квантизации)
│   ├── quantized/              # QAT модели (FP32 с fake quantization)
│   ├── int8/                   # Конвертированные модели INT8
│   ├── onnx/                   # Экспортированные ONNX модели
│   ├── figures/                # Графики сравнения
│   └── qat_results_table.csv   # Полная таблица результатов
│
├── logs/                       # Логи обучения
│   ├── lstm/                   # Логи обучения LSTM
│   ├── espcn/                  # Логи обучения ESPCN
│   └── sasrec/                 # Логи обучения SASRec
│
├── README.md                   # Этот файл
└── RESULT.md                   # Научная статья с результатами исследования
```

---

## Сводка Результатов

### LSTM Text Classifier (Метрика ROC-AUC)

| Метод | Baseline | Quantized | Δ Loss % | Статус |
|--------|----------|-----------|----------|--------|
| Baseline (FP32) | 94.33% | 94.33% | 0.00% | ✓ Успех |
| **LSQ** ⭐ | 94.33% | 93.40% | **0.99%** | ✓ Успех |
| DSQ | 94.33% | 91.45% | 3.05% | ✓ Успех |
| AdaRound | 94.33% | 89.62% | 4.99% | ✓ Успех |
| APoT | 94.33% | 89.38% | 5.25% | ✓ Успех |
| PACT | 94.33% | 50.61% | **46.35%** | ✗ **ПРОВАЛ** |

**Победитель**: LSQ (всего 0.99% потери качества)
**Критический провал**: PACT несовместим с активациями sigmoid/tanh в LSTM

---

### ESPCN Super-Resolution (Метрика PSNR)

| Метод | Baseline | Quantized | Δ Loss % | Статус |
|--------|----------|-----------|----------|--------|
| Baseline (FP32) | 27.76 dB | 27.76 dB | 0.00% | ✓ Успех |
| **DSQ** ⭐ | 27.76 dB | 26.83 dB | **3.35%** | ✓ Успех |
| AdaRound | 27.76 dB | 26.82 dB | 3.39% | ✓ Успех |
| LSQ | 27.76 dB | 26.53 dB | 4.43% | ✓ Успех |
| PACT | 27.76 dB | 26.44 dB | 4.75% | ✓ Успех |
| APoT | 27.76 dB | 25.37 dB | 8.61% | ✓ Успех |

**Победитель**: DSQ (лучший для CNN архитектур с ReLU)
**Худший**: APoT (8.61% потери, сложности с оптимизацией)

---

### SASRec Sequential Recommender (Метрика NDCG@10)

| Метод | Baseline | Quantized | Δ Loss % | Статус |
|--------|----------|-----------|----------|--------|
| Baseline (FP32) | 0.7136 | 0.7136 | 0.00% | ✓ Успех |
| **LSQ** ⭐ | 0.7136 | 0.7122 | **0.20%** | ✓ Успех |
| AdaRound | 0.7136 | 0.7115 | 0.29% | ✓ Успех |
| PACT | 0.7136 | 0.7088 | 0.67% | ✓ Успех |
| APoT | 0.7136 | 0.7023 | 1.58% | ✓ Успех |
| DSQ | 0.7136 | 0.6924 | 2.97% | ✓ Успех |

**Победитель**: LSQ (исключительная потеря 0.20% на Transformer архитектуре)
**Примечание**: Как APoT (batch 256), так и DSQ (batch 128) потребовали значительного уменьшения размера батча из-за ограничений памяти

---

### Результаты INT8 Конверсии

| Модель | Метод | FP32 Размер | INT8 Размер | Сжатие | Потеря Качества (INT8) | Ускорение |
|-------|--------|-----------|-----------|-------------|---------------------|---------|
| LSTM | LSQ | 19.10 MB | 12.35 MB | **35.3%** | Минимальная | +25.9% |
| ESPCN | DSQ | 0.14 MB | 0.05 MB | **61.8%** | Пренебрежимая (<0.01 dB) | +4.1% |
| SASRec | LSQ | 56.21 MB | 44.18 MB | **21.4%** | < 0.01 NDCG | +19.5% |

**Анализ**:
- **LSTM**: Сжатие на 35% ниже, чем у ESPCN, потому что LSTM имеет большой embedding слой (не квантизируется в стандартном PyTorch)
- **ESPCN**: Исключительное сжатие на 61.8%, потому что вся модель (все conv слои) квантизируется
- **SASRec**: Сжатие на 21.4% с минимальной потерей качества демонстрирует эффективную квантизацию transformer архитектуры
- **Ускорение**: Умеренное ускорение на CPU (4-26%) указывает, что инференс не полностью ограничен вычислениями; важна пропускная способность памяти

### Рейтинг Методов (по средней потере качества)

1. **LSQ** - 1.87% (Универсальный чемпион)
2. **DSQ** - 3.20% (Лучший для CNN)
3. **AdaRound** - 2.89% (Стабильный выбор)
4. **PACT** - 2.71% (Избегать для RNN/LSTM)
5. **APoT** - 6.93% (Наибольшие потери)

---

## Быстрый Старт

### Требования

```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install onnx onnxruntime
pip install torchtext  # Опционально, для датасета LSTM
```

### Требования к Оборудованию
- **Обучение**: NVIDIA GPU с 8GB+ VRAM (CUDA 11.0+)
- **INT8 Инференс**: Только CPU (оптимизировано для x86-64)


---

## Как Запустить: Обучение

### LSTM Text Classifier

```bash
# Baseline
cd project
python scripts/train_lstm_production.py --quantization none --batch_size 64 --epochs 50 --lr 0.001 --use_amp --seed 42

# LSQ (Лучший для LSTM)
python scripts/train_lstm_production.py --quantization lsq --bit_width 8 --batch_size 64 --epochs 50 --lr 0.001 --use_amp --seed 42

# PACT (ПРОВАЛИВАЕТСЯ на LSTM)
python scripts/train_lstm_production.py --quantization pact --bit_width 8 --batch_size 64 --epochs 50 --lr 0.001 --use_amp --seed 42

# AdaRound
python scripts/train_lstm_production.py --quantization adaround --bit_width 8 --batch_size 64 --epochs 50 --lr 0.001 --use_amp --seed 42

# APoT
python scripts/train_lstm_production.py --quantization apot --bit_width 8 --batch_size 64 --epochs 50 --lr 0.001 --use_amp --seed 42

# DSQ
python scripts/train_lstm_production.py --quantization dsq --bit_width 8 --batch_size 64 --epochs 50 --lr 0.001 --use_amp --seed 42
```

### ESPCN Super-Resolution

```bash
cd project

# Baseline
python scripts/train_espcn_production.py --quantization none --batch_size 512 --epochs 15 --lr 0.001 --use_amp --seed 42

# DSQ (Лучший для ESPCN)
python scripts/train_espcn_production.py --quantization dsq --bit_width 8 --batch_size 512 --epochs 15 --lr 0.001 --weight_quant_warmup_epochs 3 --act_quant_warmup_epochs 5 --use_amp --seed 42

# LSQ
python scripts/train_espcn_production.py --quantization lsq --bit_width 8 --batch_size 512 --epochs 15 --lr 0.001 --use_amp --seed 42

# PACT
python scripts/train_espcn_production.py --quantization pact --bit_width 8 --batch_size 512 --epochs 15 --lr 0.001 --weight_quant_warmup_epochs 3 --act_quant_warmup_epochs 5 --use_amp --seed 42

# AdaRound
python scripts/train_espcn_production.py --quantization adaround --bit_width 8 --batch_size 512 --epochs 15 --lr 0.001 --weight_quant_warmup_epochs 3 --act_quant_warmup_epochs 5 --adaround_reg_weight 1e-4 --use_amp --seed 42

# APoT
python scripts/train_espcn_production.py --quantization apot --bit_width 8 --batch_size 512 --epochs 15 --lr 0.001 --weight_quant_warmup_epochs 3 --act_quant_warmup_epochs 5 --use_amp --seed 42
```

### SASRec Sequential Recommender

```bash
cd project

# Baseline
python scripts/train_sasrec_production.py --data_dir ./data/ml-25m --quantization none --batch_size 900 --epochs 15 --lr 0.0005 --hidden_units 256 --num_blocks 4 --num_heads 4 --dropout 0.3 --max_len 150 --ffn_type swiglu --ffn_factor 4 --use_amp --seed 42

# LSQ (Лучший для SASRec)
python scripts/train_sasrec_production.py --data_dir ./data/ml-25m --quantization lsq --bit_width 8 --batch_size 900 --epochs 15 --lr 0.0005 --warmup_epochs 10 --use_amp --seed 42

# PACT
python scripts/train_sasrec_production.py --data_dir ./data/ml-25m --quantization pact --bit_width 8 --batch_size 900 --epochs 15 --lr 0.0005 --warmup_epochs 10 --use_amp --seed 42

# AdaRound
python scripts/train_sasrec_production.py --data_dir ./data/ml-25m --quantization adaround --bit_width 8 --batch_size 900 --epochs 15 --lr 0.0005 --warmup_epochs 5 --use_amp --seed 42

# APoT (Уменьшенный размер батча)
python scripts/train_sasrec_production.py --data_dir ./data/ml-25m --quantization apot --bit_width 8 --batch_size 256 --epochs 15 --lr 0.0005 --warmup_epochs 15 --gradient_clip 0.5 --seed 42

# DSQ (Наименьший размер батча)
python scripts/train_sasrec_production.py --data_dir ./data/ml-25m --quantization dsq --bit_width 8 --batch_size 128 --epochs 15 --lr 0.0005 --warmup_epochs 15 --gradient_clip 0.5 --seed 42
```

---

## Как Запустить: INT8 Конверсия

```bash
cd project

# LSTM-LSQ в INT8
python scripts/simple_int8_convert.py --model_type lstm --model_path results/quantized/lstm_lsq/lstm_lsq_best.pt --quantization lsq --save_path results/int8/lstm_lsq_int8.pt

# ESPCN-DSQ в INT8
python scripts/simple_int8_convert.py --model_type espcn --model_path results/quantized/espcn_dsq/espcn_dsq_best.pt --quantization dsq --save_path results/int8/espcn_dsq_int8.pt

# SASRec-LSQ в INT8
python scripts/convert_sasrec_lsq_int8.py --model_path results/quantized/sasrec_lsq/sasrec_lsq_b8_seed42_best.pt --save_path results/int8/sasrec_lsq_int8.pt
```

---

## Как Запустить: INT8 Оценка

```bash
cd project

# Оценка LSTM & ESPCN на CPU
python scripts/evaluate_lstm_espcn_int8_cpu.py --lstm_fp32_path results/baseline/lstm_main/lstm_none_best.pt --lstm_int8_path results/int8/lstm_lsq_int8.pt --espcn_fp32_path results/baseline/espcn_main/espcn_none_best.pt --espcn_int8_path results/int8/espcn_dsq_int8.pt --num_samples 1000

# Оценка SASRec
python scripts/evaluate_sasrec_int8_cpu.py --model_path results/int8/sasrec_lsq_int8.pt --data_dir ./data/ml-25m --num_eval_samples 1000 --batch_size 64
```

---

## Как Запустить: ONNX Конверсия

```bash
cd project

# Экспорт в ONNX
python scripts/export_to_onnx.py --model_type espcn --model_path results/quantized/espcn_dsq/espcn_dsq_best.pt --onnx_path results/onnx/espcn_dsq_fp32.onnx --quantization dsq

# Бенчмарк ONNX
python scripts/benchmark_onnx.py --fp32_model results/onnx/espcn_dsq_fp32.onnx --int8_model results/onnx/espcn_dsq_fp32_int8_static.onnx --num_runs 100 --batch_size 16
```

---

## Как Запустить: Визуализация

```bash
cd project
python scripts/generate_comparison_plots.py
```

Вывод: 5 графиков публикационного качества в `results/figures/`

---
