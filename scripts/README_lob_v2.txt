LOB v2 — этап 1

Файл:
  scripts/33_net_alpha_v2.py

Установка:
1. Скачайте архив из ChatGPT.
2. Через WinSCP загрузите 33_net_alpha_v2.py в:
   /root/rl-trading-tbank/scripts/33_net_alpha_v2.py
3. На сервере:
   cd /root/rl-trading-tbank
   chmod +x scripts/33_net_alpha_v2.py
   .venv/bin/python -m py_compile scripts/33_net_alpha_v2.py
   .venv/bin/python scripts/33_net_alpha_v2.py --help

Первый запуск в tmux:
   tmux new-session -d -s lobv2 'cd /root/rl-trading-tbank && PYTHONUNBUFFERED=1 .venv/bin/python scripts/33_net_alpha_v2.py --lob-dir data_cache/lob_test5 --horizon 15 --max-spread 2 --side-cost-bp 7 --buffer-bp 2 --rebuild 2>&1 | tee /root/lob_v2_run.txt'

Просмотр:
   tmux attach -t lobv2

Отсоединение без остановки:
   Ctrl+B, затем D

Результат:
   tail -150 /root/lob_v2_run.txt

Важно:
- Скрипт не отправляет заявки.
- Текущий период уже исследовательский. Результат нужен для выбора направления,
  а подтверждать стратегию придётся на новых днях.
