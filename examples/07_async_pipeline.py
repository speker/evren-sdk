"""
Asenkron paralel cikarim — asyncio.gather ile ayni anda birden fazla istek.

    python 07_async_pipeline.py
"""

import asyncio
from pathlib import Path
from evren_sdk import AsyncEvrenClient


async def infer(client, img, model):
    r = await client.predict(model, img, confidence=0.25)
    return Path(img).name, r


async def main():
    imgs = [f"img_{i}.jpg" for i in range(1, 6)]
    mdl = "kullanici/smoke-detection"

    async with AsyncEvrenClient(api_key="evren_...") as c:
        await c.warmup([mdl])

        jobs = [infer(c, img, mdl) for img in imgs]
        for item in await asyncio.gather(*jobs, return_exceptions=True):
            if isinstance(item, Exception):
                print(f"  HATA: {item}")
            else:
                name, r = item
                print(f"  {name}: {r.count} tespit {r.inference_ms:.0f}ms")

        # batch alternatifi
        batch = await c.predict_batch(mdl, imgs[:3])
        print(f"\nbatch: {batch.count} gorsel {batch.total_ms:.0f}ms")


asyncio.run(main())
