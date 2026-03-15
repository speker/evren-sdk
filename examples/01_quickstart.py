"""
Basit tekil cikarim.

    pip install evren-sdk
    python 01_quickstart.py
"""

from evren_sdk import EvrenClient

client = EvrenClient(api_key="evren_YOUR_KEY_HERE")

result = client.predict(
    model="kullanici/smoke-detection",
    image="test.jpg",
    confidence=0.3,
)

print(f"{result.count} nesne bulundu ({result.inference_ms:.0f}ms)")
for det in result.predictions:
    print(f"  {det.class_name}: {det.confidence:.0%}  bbox={det.bbox}")

# bytes da kabul eder
raw = open("test.jpg", "rb").read()
r2 = client.predict("kullanici/smoke-detection", raw)
print(f"\nbytes: {r2.count} nesne")
