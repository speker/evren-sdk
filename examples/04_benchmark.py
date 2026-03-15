# Model gecikme & throughput olcumu
#
#   python 04_benchmark.py

from evren_sdk import EvrenClient

client = EvrenClient(api_key="evren_...")

b = client.benchmark(
    model="kullanici/smoke-detection",
    image="test.jpg",
    rounds=20,
    warmup_rounds=3,
)

print(f"avg={b.avg_ms:.1f}ms  min={b.min_ms:.1f}ms  max={b.max_ms:.1f}ms  p95={b.p95_ms:.1f}ms")
print(f"throughput: {b.throughput_fps:.1f} FPS  ({b.rounds} round)")
