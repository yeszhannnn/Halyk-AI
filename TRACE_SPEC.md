# trace.json — отладочный формат

Переписано 6 августа. Предыдущая версия (полный W3C PROV + Web Annotation +
XBRL OIM) отложена: организатор подтвердил, что смотрят пайплайн, а
доказательный слой баллов не приносит. Раздел 5 сохраняет её на после 9 августа.

**Назначение сейчас — отладка.** Единственный контур обратной связи это
локальный скорер, а он говорит только «эта ячейка неверна». Трейс отвечает
на вопрос «почему», и от скорости этого ответа зависит, сколько итераций
влезет между 11:00 и 13:30.

---

## 1. Верхний уровень

```json
{
  "run":      { ... },
  "cells":    [ ... ],
  "conflicts":[ ... ],
  "review":   [ ... ]
}
```

`submission.json` — чистая проекция `cells`. Никаких вызовов моделей
при пересборке: изменились требования к формату — правишь `project()`
и собираешь заново за секунду.

---

## 2. Ячейка

Одна запись на каждую из 36 ячеек.

```json
{
  "scenario_id": "P8",
  "slot": "6.1",

  "status": "BREACH",
  "actual": 4221314.95,
  "evidence_txn_id": null,

  "covenant": {
    "title": "Максимальные совокупные обязательства по персоналу",
    "direction": "MAX",
    "threshold": "4000000.00",
    "period": ["2025-01-01", "2025-12-31"],
    "springing": null,
    "source": { "doc": "3f9a2c81b70e.pdf", "page": 5,
                "quote": "не должны превышать $4,000,000.00" }
  },

  "computation": {
    "expression": "SUM(personnel_liabilities)",
    "ledger_component": "3302867.43",
    "adjustments_applied": ["adj_p8_7_1"],
    "off_ledger_added": "918447.52",
    "evaluated": "4221314.95",
    "rounded": "4221314.95",
    "comparison": "4221314.95 > 4000000.00 → BREACH"
  },

  "inputs": [
    { "txn_id": "TXN-P8-0003", "amount_usd": "-412663.28",
      "category": "personnel", "why": "description~payroll" }
  ],

  "evidence_search": {
    "method": "counterfactual",
    "candidates_tested": 14,
    "flipping": [],
    "result": null,
    "reason": "агрегатный лимит, одна операция вердикт не переворачивает"
  },

  "confidence": 0.91,
  "strategy": "computed",
  "flags": []
}
```

**Почему именно эти поля.**

`ledger_component` отдельно от `off_ledger_added` — потому что самая частая
ошибка на этом датасете это посчитать только по CSV. Видно сразу, применилось
раскрытие или нет.

`adjustments_applied` — список идентификаторов корректировок. Пусто там, где
ожидалась корректировка, значит стадия 4c не сработала.

`evaluated` строкой и отдельно от `rounded` — обязательно. Статус считается
по неокруглённому. Доказано: P4 и P8 имеют одинаковый порог 0.04 и одинаковый
округлённый факт 0.04, но противоположные вердикты.

`inputs` с полем `why` — почему строка попала в категорию. При расхождении
с ключом это первое место, куда смотреть: категоризация промахнулась чаще,
чем арифметика.

`evidence_search` с числом кандидатов и списком переворачивающих — показывает,
контрфактуал не нашёл ничего или нашёл слишком много.

---

## 3. Корректировки и конфликты

```json
"adjustments": {
  "adj_p8_7_1": {
    "kind": "OFF_LEDGER",
    "amount": "918447.52",
    "category": "personnel",
    "source": { "doc": "a7c31f90bd42.pdf", "page": 2,
                "quote": "не отражается отдельной операцией в бухгалтерской книге" }
  },
  "adj_p2_9_1": {
    "kind": "RECLASS",
    "amount": "1104663.28",
    "counterparty": "Tien Shan Advisory Bureau",
    "from_category": "consulting",
    "to_category": "opex",
    "matched_txn": "TXN-P2-0040",
    "match_method": "amount+counterparty",
    "source": { "doc": "...", "page": 2, "quote": "переклассифицирована" }
  }
}
```

```json
"conflicts": [
  { "kind": "AMBIGUOUS_RECLASS_MATCH", "scenario": "P7",
    "adjustment": "adj_p7_9_1", "candidates": ["TXN-P7-0012", "TXN-P7-0031"] },
  { "kind": "PERIOD_MISMATCH", "doc": "c10ebf055fa5.pdf",
    "expected": "2025", "found": "2024" }
]
```

`conflicts.json` открывается первым при разборе провала. Он показывает,
где сломался парсинг, быстрее любого лога.

Типы конфликтов, которые обязаны детектироваться:
`AMBIGUOUS_RECLASS_MATCH`, `PERIOD_MISMATCH`, `UNBOUND_DOCUMENT`,
`MULTIPLE_ACTIVE_LOANS`, `KYC_THRESHOLD_NOT_FOUND`, `QUOTE_UNVERIFIED`,
`MULTIPLE_FLIPPING_TXNS`.

---

## 4. Прогон и самопроверка

```json
"run": {
  "run_id": "...", "started_at": "...", "mode": "full",
  "code": { "git_sha": "e4b71c9" },
  "inputs": { "dataset_sha256": "...", "pdf_count": 200, "ledger_rows": 1473 },
  "models": [{ "id": "gpt-4o-mini", "temperature": 0 }],
  "counters": { "classified_noise": 145, "loans_active": 12,
                "adjustments_found": 9, "quotes_rejected": 3 },
  "cost": { "usd": 0.41 }, "cache": { "hits": 812 }
}
```

Счётчики в `counters` — дешёвая проверка на здравый смысл. Если
`loans_active` не равно числу сценариев или `adjustments_found` внезапно ноль,
что-то сломалось на уровне классификации, и это видно до всякого скоринга.

`trace.verify()` проверяет:

1. Каждая `quote` — подстрока текста своей страницы.
2. `comparison` согласуется со `status`.
3. `rounded` получается из `evaluated` заявленным округлением.
4. Каждый `evidence_txn_id` существует в леджере.
5. Улика непустая ⟹ `status == "BREACH"`.
6. Каждый `adjustments_applied` разрешается в существующую корректировку.
7. Число ячеек равно числу ячеек шаблона.

---

## 5. Отложено до после 9 августа

Для портфолио, не для сдачи. Возвращается полная версия:
W3C Web Annotation с массивом селекторов и деградацией
(`TextQuoteSelector` первичный, `FragmentSelector` с bbox вторичный),
W3C PROV с графом вывода и разделением агентов, XBRL OIM для числовых фактов
(строки вместо float, instant против duration).

Это читается серьёзно для риск-команды банка и стоит времени — но после того,
как сдача улетела.
