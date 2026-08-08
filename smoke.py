"""Проверка провайдера и модели за 3 секунды и полцента."""
import sys, base64
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from agent import config
from agent.llm.client import LLMClient
from agent.llm.schemas.covenants import CovenantExtract

print(f"provider={config.LLM_PROVIDER}  model={config.MODEL_ID}\n")

import asyncio

async def main():
    c = LLMClient()

    print("1. текст + tool calling")
    try:
        r = await c.complete(
            prompt="Пункт 6.1. Отношение Долг/EBITDA не должно превышать 3.50x "
                   "в период с 2025-01-01 по 2025-12-31.",
            response_model=CovenantExtract,
        )
        print(f"   OK  threshold={r.threshold} direction={r.direction}")
    except Exception as e:
        print(f"   FAIL  {type(e).__name__}: {str(e)[:200]}")
        return

    print("2. vision")
    img = next(Path("data/rehearsal").rglob("*.png"), None)
    if not img:
        print("   пропущено: нет отрендеренной страницы")
    else:
        try:
            r = await c.complete_vision(
                prompt="Что изображено? Одно предложение.",
                image_path=str(img),
                response_model=None,
            )
            print(f"   OK  {str(r)[:80]}")
        except Exception as e:
            print(f"   FAIL  {type(e).__name__}: {str(e)[:200]}")

    print(f"\nтокены: {c.counter.total_tokens}  стоимость: ${c.counter.cost_usd}")

asyncio.run(main())