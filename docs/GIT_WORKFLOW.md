# Git Workflow и контроль версий

## Структура веток

### Главная ветка (main/master)
- **Защищена**: Требует pull request и ревью
- **Статус**: Всегда в рабочем состоянии
- **Политика слияния**: Только через pull request с минимум 1-2 ревьюерами

### Ветки разработки
- **feature/***: Новая функциональность
- **bugfix/***: Исправление ошибок
- **hotfix/***: Критические исправления
- **refactor/***: Рефакторинг кода

## Workflow

### 1. Создание ветки

```bash
# Обновить главную ветку
git checkout main
git pull origin main

# Создать новую ветку
git checkout -b feature/new-feature
```

### 2. Разработка

```bash
# Регулярные коммиты
git add .
git commit -m "feat: add new feature description"

# Push в удаленный репозиторий
git push origin feature/new-feature
```

### 3. Создание Pull Request

1. Перейти на GitHub/GitLab
2. Создать Pull Request из feature ветки в main
3. Заполнить описание изменений
4. Назначить ревьюеров (1-2 человека)
5. Дождаться одобрения

### 4. Ревью

**Критерии одобрения**:
- Код соответствует стандартам проекта
- Тесты проходят
- Документация обновлена (если нужно)
- Нет конфликтов с main

**Процесс**:
- Ревьюер проверяет код
- Оставляет комментарии (если нужно)
- Одобряет или запрашивает изменения
- После одобрения можно мержить

### 5. Слияние

```bash
# После одобрения PR, слить в main
git checkout main
git pull origin main
git merge feature/new-feature
git push origin main
```

## Правила коммитов

### Формат сообщений

Используйте [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Типы коммитов

- **feat**: Новая функциональность
- **fix**: Исправление ошибки
- **docs**: Изменения в документации
- **style**: Форматирование кода
- **refactor**: Рефакторинг
- **test**: Добавление/изменение тестов
- **chore**: Изменения в конфигурации, зависимостях

### Примеры

```bash
git commit -m "feat(api): add rate limiting to OpenAQ client"
git commit -m "fix(etl): handle missing weather data gracefully"
git commit -m "docs: update README with installation instructions"
git commit -m "test(quality): add tests for data validation"
```

## Защита главной ветки

### Настройки (GitHub)

1. Перейти в Settings → Branches
2. Добавить правило для `main`:
   - ✅ Require pull request reviews before merging
   - ✅ Required number of reviewers: 1-2
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

### Настройки (GitLab)

1. Перейти в Settings → Repository → Protected branches
2. Защитить `main`:
   - Allowed to merge: Maintainers
   - Allowed to push: No one
   - Allowed to force push: No
   - Require approval from code owners: Yes

## CI/CD интеграция

### GitHub Actions

```yaml
name: CI
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest --cov=src
      - run: flake8 src tests
      - run: mypy src
```

### GitLab CI

```yaml
test:
  script:
    - pip install -r requirements.txt
    - pytest --cov=src
    - flake8 src tests
    - mypy src
  only:
    - merge_requests
    - main
```

## Ревью кода

### Чеклист для ревьюера

- [ ] Код соответствует стилю проекта
- [ ] Нет очевидных ошибок
- [ ] Тесты добавлены/обновлены
- [ ] Документация обновлена
- [ ] Нет хардкода секретов
- [ ] Обработка ошибок присутствует
- [ ] Логирование добавлено где нужно

### Комментарии

Используйте конструктивные комментарии:
- ✅ "Можно улучшить: ..."
- ✅ "Предложение: ..."
- ❌ Избегайте: "Это плохо", "Неправильно"

## История коммитов

### Регулярность

- Коммиты после завершения конкретной задачи
- Коммиты после добавления новой функциональности
- Коммиты после исправления ошибки

### Размер коммитов

- Один коммит = одна логическая единица работы
- Не смешивать несвязанные изменения
- Использовать `git add -p` для выборочного добавления

## Теги версий

### Семантическое версионирование

```
v<major>.<minor>.<patch>
```

- **major**: Несовместимые изменения API
- **minor**: Новая функциональность (обратно совместимая)
- **patch**: Исправления ошибок

### Создание тега

```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

## Резюме

1. **Главная ветка защищена**: Требует PR и ревью
2. **Регулярные коммиты**: После каждой завершенной задачи
3. **Понятные сообщения**: Используйте Conventional Commits
4. **Ревью перед слиянием**: Минимум 1-2 ревьюера
5. **CI/CD проверки**: Автоматические тесты перед слиянием

