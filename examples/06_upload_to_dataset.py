# SDK uzerinden veri setine gorsel yukle.
# Duplikat gorseller otomatik algilanir.
#
#   python 06_upload_to_dataset.py images/*.jpg

import sys
from pathlib import Path
from evren_sdk import EvrenClient

client = EvrenClient(api_key="evren_xxxxx")

DS = "019ce417-..."  # veri seti UUID

for img in sys.argv[1:] or ["sahne.jpg"]:
    resp = client.upload_to_dataset(DS, img)
    if resp.get("duplicate"):
        print(f"  {Path(img).name}: zaten var")
    else:
        print(f"  {Path(img).name}: yuklendi")
