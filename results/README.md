# Результаты, включённые в репозиторий

В эту папку включены только небольшие артефакты, достаточные для проверки ключевых выводов. Большие промежуточные датасеты и runtime-state исключены.

## Что здесь есть

- `json/ofi_results.json` — актуальная сводка OFI/microstructure эксперимента.
- `json/mr30_v3_frozen.json` — замороженная спецификация MR30 до forward-проверки.
- `json/forward_eval_multiday_summary.json` — результат раннего forward после freeze.
- `tables/mr30_exact_ablation_leaderboard.csv` — ablation базового MR-сигнала.
- `sandbox/mr30_sandbox_journal.csv` — журнал сделок брокерской песочницы.
- `tables/triple_barrier_cost_frontier.csv` — чувствительность triple-barrier стратегии к стоимости исполнения.
- `tables/exact_cost_candidate_summary.csv` — точный cost robustness для двух кандидатов.
- `tables/regime_flip_results.csv` — проверка regime-flip.
- `tables/flow_momentum_results.csv` — независимая flow-confirmed momentum ветка.

`*_best_trade_examples.csv` и `*_worst_trade_examples.csv` — только иллюстрации отдельных сделок для разбора; они не используются как оценка общей доходности.
