"""Sonuclari filtrele, YOLO/COCO/CSV/JSON olarak kaydet."""

from evren_sdk import EvrenClient

client = EvrenClient(api_key="evren_xxxxx")

result = client.predict("kullanici/model", "sahne.jpg", confidence=0.1)
print(f"Ham: {result.count} tespit\n")

# filtrele
yuksek = result.filter(min_confidence=0.7)
print(f"conf >= 0.7 : {yuksek.count}")

sadece_araba = result.filter(classes=["araba"])
print(f"araba       : {sadece_araba.count}")

combo = result.filter(min_confidence=0.5, max_confidence=0.9, classes=["insan"])
print(f"insan 0.5-0.9: {combo.count}")

print("\n-- YOLO --")
print(result.to_yolo()[:200])

print("\n-- COCO --")
for ann in result.to_coco()[:3]:
    print(f"  cat={ann['category_id']}  bbox={ann['bbox']}  area={ann['area']:.0f}")

print("\n-- CSV --")
print(result.to_csv()[:300])

# uzantiya gore format secer
result.save("output/sonuc.json")
result.save("output/sonuc.csv")
result.save("output/labels.txt")
