# SDK uzerinden veri seti olustur ve dosya yukle (gorsel veya NLP/tablo).
# Duplikat dosyalar otomatik algilanir.
#
#   python 06_upload_to_dataset.py images/*.jpg
#   python 06_upload_to_dataset.py --new "benim-veri-setim" --modality NLP verses.jsonl

import sys
from pathlib import Path
from evren_sdk import EvrenClient

client = EvrenClient(api_key="evren_plt_xxxxx")

if sys.argv[1:2] == ["--new"]:
    name, modality, files = sys.argv[2], sys.argv[4], sys.argv[5:]
    ds = client.create_dataset(name, modality=modality, visibility="public")
    DS = ds["id"]
    print(f"veri seti olusturuldu: {ds['slug']} ({DS})")
else:
    DS = "019ce417-..."  # var olan veri seti UUID
    files = sys.argv[1:] or ["sahne.jpg"]

for f in files:
    resp = client.upload_to_dataset(DS, f)
    if resp.get("duplicate"):
        print(f"  {Path(f).name}: zaten var")
    else:
        print(f"  {Path(f).name}: yuklendi")
