# AI Designer — CrewAI Flow

Мультиагентная дизайн-система на **CrewAI Flow 1.10+**.
Агенты работают через typed `FlowState`, шаги соединены `@start` / `@listen` / `@router`.

## Архитектура Flow

```
@start  research()         — Research Agent (Firecrawl + WebSearch)
          │
@listen analyze_style()    — Style Analyst → DesignBrief (JSON)
          │
@listen generate()         — Design Generator → файл в output/
          │
@listen critique()         — Critic Agent → CritiqueResult (JSON)
          │
@router check_quality()    ──→ "done"     (score ≥ MIN_SCORE)
                           └──→ "generate" (итерация, score < MIN_SCORE)
          │
@listen done()             — финальный вывод
```

Весь shared state — `DesignerState(FlowState)`. Каждый шаг читает и пишет
в `self.state` напрямую. Никаких `context=[]` между тасками — данные
передаются через state.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # заполни ключи
```

`.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
FIRECRAWL_API_KEY=fc-...
MIN_QUALITY_SCORE=7.0
OUTPUT_DIR=./output
```

## Использование

```bash
python main.py "лэндинг для архитектурного бюро"
python main.py "фирменный стиль ресторана" --format brandbook
python main.py "сайт для фотографа" --format moodboard
python main.py "премиальный ювелирный бренд" --format html --min-score 8
```

## Форматы

| Флаг | Файл | Описание |
|------|------|----------|
| `html` | `landing.html` | Полный одностраничный лэндинг |
| `svg` | `layout.svg` | Визуальный макет 1440×900px |
| `moodboard` | `moodboard.html` | Референсы, палитра, типографика |
| `brandbook` | `brandbook.html` | Логотип, цвета, шрифты, примеры |

## Структура проекта

```
ai-designer/
├── main.py               # CLI (click)
├── flow.py               # DesignerFlow — весь пайплайн
├── agents/
│   └── builders.py       # Фабрики агентов и тасков
├── tools/
│   └── design_tools.py   # Firecrawl, WebSearch, FileWriter
├── models/
│   └── state.py          # DesignerState(FlowState), DesignBrief, CritiqueResult
└── output/               # Результаты генерации
```

## Как добавить новый формат

1. `models/state.py` — добавь значение в `OutputFormat`
2. `agents/builders.py` — добавь инструкцию в `_FORMAT_INSTRUCTIONS`
3. `main.py` — добавь в `click.Choice`