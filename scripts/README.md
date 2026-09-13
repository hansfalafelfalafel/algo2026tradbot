# Карта экспериментов

Нумерация сохранена специально: она отражает хронологию исследования и позволяет воспроизводить отдельные этапы без переписывания путей между скриптами.

| Этап | Скрипты | Содержание |
|---|---|---|
| Базовый pipeline | `00–11` | поиск инструмента, загрузка данных, RL-обучение, backtest, sandbox, ансамбль, walk-forward |
| Альтернативные стратегии | `20–26` | пары, факторный momentum, meta-labeling, long/short |
| Микроструктура | `30–42` | сбор стакана, OFI, execution simulator, cost-aware alpha, AlgoPack, DL/GRU |
| Честный forward | `43–54` | forward evaluation, диагностика провала, freeze стратегии, regime research, shadow |
| Broker sandbox | `55–58` | исполнение MR30 через песочницу, статистика, решения, Telegram |
| Robustness / новые гипотезы | `59–73` | strategy lab, walk-forward, regime/cross-sectional, meta-gates, triple barrier, cost frontier, regime flip, flow momentum |

Backup/corrupt-файлы и `__pycache__` из рабочей директории удалены из сдаваемой версии.
