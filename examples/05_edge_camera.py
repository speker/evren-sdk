"""
Edge cihazda gercek zamanli cikarim — GPU gerektirmez.
Cikarim EVREN GPU'larinda calisir, lokal frame'e cizilir.

    pip install evren-sdk[edge]
    python 05_edge_camera.py
"""

from evren_sdk import EvrenCamera

cam = EvrenCamera("evren_...", "kullanici/smoke-detection", confidence=0.3, max_fps=15)

# webcam ac, ESC ile kapat
cam.run(0)

# -- alternatifler (birini uncomment edin) --

# kendi loop'unuzda
# for frame, result in cam.stream(0):
#     print(f"{result.count} tespit")

# video isle + kaydet
# cam.record("input.mp4", "output.mp4")

# klasor tara
# for path, result in cam.scan("images/", save_to="results/"):
#     print(f"{path.name}: {result.count}")

# RTSP
# for frame, result in cam.stream("rtsp://admin:pass@192.168.1.10/stream"):
#     pass
