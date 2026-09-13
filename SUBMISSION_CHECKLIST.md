# Чек-лист перед сдачей

- [ ] Скопировать `.env.example` в `.env` только локально и заполнить токены.
- [ ] Никогда не коммитить `.env`, API-токены, sandbox account id и локальный `state/`.
- [ ] Проверить `python -m compileall -q src scripts tests dashboard`.
- [ ] Установить dev-зависимости и запустить `pytest`, `flake8`, `pylint` командами из `docs/README_DEV.md`.
- [ ] Загрузить содержимое этой папки в `hansfalafelfalafel/algo2026tradbot`.
- [ ] Убедиться, что GitHub показывает root `README.md`, Mermaid-схему и `results/`.
- [ ] После загрузки проверить, что в репозитории нет `data_cache/`, `state/`, `.env`, `*.log`, backup/corrupt файлов.
- [ ] Отправить ссылку на репозиторий и презентацию до дедлайна.

Презентация будет собрана отдельно на основе `README.md`, `results/` и схемы архитектуры.
