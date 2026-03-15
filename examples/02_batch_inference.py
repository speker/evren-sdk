# Toplu cikarim — CLI'dan gorsel listesi alir
#
#   python 02_batch_inference.py images/*.jpg

import sys
from pathlib import Path
from evren_sdk import EvrenClient

client = EvrenClient(api_key="evren_...")

imgs = sys.argv[1:] or ["img1.jpg", "img2.jpg", "img3.jpg"]

batch = client.predict_batch("kullanici/smoke-detection", imgs, confidence=0.25, iou=0.45)

for i, r in enumerate(batch):
    name = Path(imgs[i]).name if i < len(imgs) else f"#{i}"
    print(f"  {name}: {r.count} nesne  {r.inference_ms:.0f}ms")

print(f"\n{batch.count} gorsel toplam {batch.total_ms:.0f}ms")
