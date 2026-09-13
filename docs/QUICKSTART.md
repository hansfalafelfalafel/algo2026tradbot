# Быстрый старт

Ниже — минимальная инструкция для запуска проекта на обычном компьютере. Для обучения, backtest и разовых экспериментов отдельный сервер не нужен. Сервер имеет смысл только для процессов, которые должны работать постоянно.

## 1. Установка

Проверьте версию Python:

```bash
python --version
```

Проект рассчитан на Python 3.9–3.11.

Перейдите в папку проекта:

```bash
cd rl-trading-tbank
```

Создайте виртуальное окружение:

```bash
python -m venv .venv
```

Активация в Windows:

```bash
.venv\Scripts\activate
```

Активация в macOS / Linux:

```bash
source .venv/bin/activate
```

Установите зависимости:

```bash
pip install -r requirements.txt
```

SDK T-Bank устанавливается отдельно:

```bash
pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
```

## 2. Токен T-Bank Invest API

Создайте локальный `.env` на основе шаблона:

```bash
cp .env.example .env
```

Для Windows:

```bash
copy .env.example .env
```

В `.env` укажите токен:

```text
TINKOFF_TOKEN=...
```

`.env` не должен попадать в Git.

## 3. Базовый pipeline

Самый короткий путь от данных до backtest:

```bash
python scripts/01_download_data.py
python scripts/02_train.py
python scripts/03_backtest.py
```

Для портфельной RL-ветки:

```bash
python scripts/06_train_ensemble.py
python scripts/07_backtest_portfolio.py
```

Для walk-forward проверки:

```bash
python scripts/11_walkforward.py --folds 4
```

Поздние эксперименты с микроструктурой используют отдельный локальный кэш стакана и сделок, поэтому не все скрипты `30–73` можно воспроизвести только на маленьких примерах из репозитория.

## 4. Новости, рекомендации и Telegram

Новостной сигнал:

```bash
python scripts/05_fetch_news.py
```

Тестовый запуск:

```bash
python scripts/05_fetch_news.py --sample
```

Получение рекомендации:

```bash
python scripts/08_recommend.py
```

Для Telegram в `.env` используются:

```text
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## 5. Дашборд

```bash
streamlit run dashboard/app.py
```

После запуска Streamlit покажет локальный адрес, обычно `http://localhost:8501`.

## 6. Песочница

Для проверки торговой логики без реальных денег используется T-Bank Sandbox.

```bash
python scripts/09_run_live.py
```

Дополнительные режимы:

```bash
python scripts/09_run_live.py --stream
python scripts/09_run_live.py --once
```

Перед любым использованием реального счёта стратегию нужно отдельно проверять на свежих данных и в песочнице. Результаты проекта не подтверждают наличие устойчиво прибыльной стратегии для реальных денег.

## 7. Для первых воспроизводимых результатов

Для демонстрации основного воспроизводимого pipeline достаточно:

```bash
python scripts/01_download_data.py
python scripts/02_train.py
python scripts/03_backtest.py
```

Хронология исследования описана в `scripts/README.md`, а сохранённые результаты — в `results/README.md`.

## 8. Компьютер или сервер

Для обучения, теста на исторических данных и анализа достаточно обычного компьютера.

Сервер нужен только для процессов, которые должны работать без остановки: сборщика стакана, наблюдения без заключения сделок или просмотра дашборда.
