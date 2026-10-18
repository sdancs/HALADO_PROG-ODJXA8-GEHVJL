# Kényelmesen tudunk fájl- és mappaútvonalakat kezelni
from pathlib import Path

# A pillow, pil könyvtárból importáljuk az Image osztályt
from PIL import Image, UnidentifiedImageError

# Az ultralytics könyvtárból importáljuk a YOLO modellt
from ultralytics import YOLO


# Meghatározzuk a projekt főmappájának elérési útját
# A __file__ az aktuális Python-fájl helyére mutat
PROJECT_DIR = Path(__file__).resolve().parent

# Összeállítjuk a tesztkép teljes elérési útját, projektmappába keresünk
IMAGE_PATH = PROJECT_DIR / "adatok" / "felulnezet.jpg"

# A használni kívánt előre betanított YOLO modell neve
MODEL_NAME = "yolov8n.pt"

# A COCO adathalmazban szereplő jármű kategóriák azonosítói (ID) és nevei
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def load_image(image_path):
    # Ellenőrizzük, hogy a megadott útvonalon létezik-e fájl
    if not image_path.is_file():
        raise FileNotFoundError(
            # A hibaüzenetben megjelenítjük a keresett fájl útvonalát:
            f"A képfájl nem található: {image_path}"
        )

    # A try blokkban olyan műveleteket végzünk, amivel azt ellenőrizzük, hogy a kép kép-e
    try:
        # A with gondoskodik arról, hogy a fájl megfelelően bezáródjon.
        with Image.open(image_path) as image:
            # Ellenőrizzük a kép fájlszerkezetét, de be is zárja
            image.verify()
        with Image.open(image_path) as image:
            # ha olvasási hiba van, nem nyílik meg
            image.load()
            # fontos adatokat eltároljuk egy szótárba:
            image_info = {
                "format": image.format,
                "size": image.size,
                "mode": image.mode,
            }

        # Visszaadjuk a kép tulajdonságait tartalmazó szótárat:
        return image_info

    # Kezeljük azokat a hibákat, ha előfordultak:
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"A fájl nem olvasható érvényes képként: {image_path}"
        ) from error


def detect_vehicles(image_path, model_name=MODEL_NAME):
    # Betöltjük az előre betanított YOLO modellt (első futáskor letölti a fájlt)
    model = YOLO(model_name)

    # Lefuttatjuk a felismerést a képen (verbose=False elrejti a felesleges rendszerüzeneteket)
    results = model(str(image_path), verbose=False)
    result = results[0]

    detected_vehicles = []

    # Végigmegyünk a képen talált összes objektum határolókeretén (bounding box)
    for box in result.boxes:
        class_id = int(box.cls[0].item())

        # Csak a járműveket tartjuk meg a szótárunk alapján
        if class_id in VEHICLE_CLASSES:
            confidence = float(box.conf[0].item())
            # A befoglaló téglalap koordinátái: bal felső (x1, y1) és jobb alsó (x2, y2) sarok
            x1, y1, x2, y2 = [round(coord) for coord in box.xyxy[0].tolist()]

            detected_vehicles.append({
                "label": VEHICLE_CLASSES[class_id],
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
            })

    return detected_vehicles


def main():
    print("PARKOLÓHELY FELISMERÉS")
    print("-" * 35)

    # A fent megírt függvénnyel meghívjuk a képet, eltároljuk
    try:
        image_info = load_image(IMAGE_PATH)
        # Jelezzük, hogy a kép betöltése és ellenőrzése sikeres volt, mivel nem volt a függvényben hiba
        print("A kép sikeresen betöltve.")
        print(f"Formátum: {image_info['format']}")
        print(f"Felbontás: {image_info['size'][0]} x {image_info['size'][1]} pixel")
        print(f"Színmód: {image_info['mode']}")
        print("-" * 35)

        # Járművek felismerésének indítása
        print("Járművek keresése a képen...")
        vehicles = detect_vehicles(IMAGE_PATH)

        print(f"Felismert járművek száma: {len(vehicles)} db\n")

        # Eredmények részletes kiírása a konzolra
        for index, vehicle in enumerate(vehicles, start=1):
            label = vehicle["label"]
            conf_percent = vehicle["confidence"] * 100
            x1, y1, x2, y2 = vehicle["bbox"]
            print(
                f"{index}. Típus: {label} | "
                f"Bizonyosság: {conf_percent:.1f}% | "
                f"Hely (x1, y1, x2, y2): ({x1}, {y1}) - ({x2}, {y2})"
            )

    # Kivétel kezelés, ha a load_image() függvény ide dobja
    except (FileNotFoundError, ValueError) as error:
        print(f"Hiba: {error}")


if __name__ == "__main__":
    main()