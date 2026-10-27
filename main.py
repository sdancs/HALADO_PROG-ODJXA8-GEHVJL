
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from ultralytics import YOLO


# A projekt főmappája
PROJECT_DIR = Path(__file__).resolve().parent

# Bemeneti és kimeneti képek
IMAGE_PATH = PROJECT_DIR / "adatok" / "parhuzamos.jpg"
OUTPUT_PATH = PROJECT_DIR / "adatok" / "autok_eredmeny.jpg"

# Előre betanított YOLO modell
MODEL_NAME = "yolov8n.pt"

# Felismerendő járműkategóriák
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# A járműkeretek oldalankénti csökkentésének aránya, később kell majd a vonalakhoz
BOX_SHRINK_RATIO = 0.20



def load_image(image_path):
    # Ellenőrizzük a fájl létezését
    if not image_path.is_file():
        raise FileNotFoundError(
            f"A képfájl nem található: {image_path}"
        )

    # Ellenőrizzük, hogy a fájl érvényes kép-e
    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            image.load()

            return {
                "format": image.format,
                "size": image.size,
                "mode": image.mode,
            }

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"A fájl nem olvasható érvényes képként: {image_path}"
        ) from error


# A YOLO által meghatározott keret összehúzása
def shrink_bbox(bbox, image_width, image_height):
    # A keret koordinátái
    x1, y1, x2, y2 = bbox

    # A keret szélessége és magassága
    width = x2 - x1
    height = y2 - y1

    # A levágás mértéke oldalanként
    cut_x = width * BOX_SHRINK_RATIO
    cut_y = height * BOX_SHRINK_RATIO

    # A keret összehúzása
    x1 = round(x1 + cut_x)
    y1 = round(y1 + cut_y)
    x2 = round(x2 - cut_x)
    y2 = round(y2 - cut_y)

    # A koordinátákat a kép határain belül tartjuk
    x1 = max(0, min(x1, image_width - 1)) #középre mutasson
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))

    return x1, y1, x2, y2



def detect_vehicles(image_path, model_name=MODEL_NAME):
    # Betöltjük a YOLO modellt
    model = YOLO(model_name)

    # Lefuttatjuk a járműfelismerést
    results = model(str(image_path), verbose=False)
    result = results[0]

    # A felismert járművek listája
    detected_vehicles = []

    # Lekérjük a kép méretét
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    # Végigmegyünk a felismert objektumokon
    for box in result.boxes:
        # A kategóriaazonosító
        class_id = int(box.cls[0].item())

        # Csak a járműkategóriákat tartjuk meg
        if class_id in VEHICLE_CLASSES:
            # A felismerés bizonyossága
            confidence = float(box.conf[0].item())

            # Az eredeti keret koordinátái
            original_bbox = tuple(
                round(coord)
                for coord in box.xyxy[0].tolist()
            )

            # A keret összehúzása
            bbox = shrink_bbox(
                original_bbox,
                image_width,
                image_height
            )

            # A jármű adatainak eltárolása
            detected_vehicles.append({
                "label": VEHICLE_CLASSES[class_id],
                "confidence": confidence,
                "bbox": bbox,
            })

    # Betöltjük az eredeti képet RGB formátumban
    with Image.open(image_path) as image:
        annotated_image = image.convert("RGB") #feliraots kép

    # Létrehozzuk a rajzoló objektumot
    draw = ImageDraw.Draw(annotated_image)

    # A betűméretet a kép szélességéhez igazítjuk
    font_size = max(16, annotated_image.width // 50)

    # Betűtípus betöltése
    font = ImageFont.truetype("arial.ttf", font_size)

    # Kirajzoljuk a járművek csökkentett kereteit:
    for vehicle in detected_vehicles:
        x1, y1, x2, y2 = vehicle["bbox"]

        # A jármű címkéje
        text = (
            f"{vehicle['label']} "
            f"{vehicle['confidence']:.0%}"
        )

        # Zöld keret a jármű körül
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(0, 255, 0),
            width=max(2, annotated_image.width // 400),
        )

        # A címke mérete
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # A címkét a keret fölé helyezzük
        text_y = max(0, y1 - text_height - 8)

        # Háttér a szöveg mögött legyen
        draw.rectangle(
            (
                x1,
                text_y,
                x1 + text_width + 8,
                text_y + text_height + 6,
            ),
            fill=(0, 180, 0),
        )

        # A címke kiírása
        draw.text(
            (x1 + 4, text_y + 3),
            text,
            fill=(255, 255, 255),
            font=font,
        )

    # Az eredménykép mentése
    annotated_image.save(
        OUTPUT_PATH,
        format="JPEG",
        quality=100,
    )

    # Eredmények kiírása
    print(f"\nEredménykép elmentve: {OUTPUT_PATH}")
    print(f"Felismert járművek: {len(detected_vehicles)} db")

    return detected_vehicles


def main():
    print("YOLO JÁRMŰFELISMERÉS")
    print("-" * 35)

    try:
        # Ellenőrizzük a bemeneti képet
        image_info = load_image(IMAGE_PATH)

        # Kiírjuk a kép adatait
        print("A kép sikeresen betöltve.")
        print(f"Formátum: {image_info['format']}")
        print(f"Felbontás: {image_info['size'][0]} x {image_info['size'][1]} pixel")
        print(f"Színmód: {image_info['mode']}")
        print("-" * 35)

        # Elindítjuk a felismerést
        vehicles = detect_vehicles(IMAGE_PATH)
        # Részletes eredmények kiírása
        print("\nJÁRMŰVEK ADATAI")
        for index, vehicle in enumerate(vehicles, start=1):
            x1, y1, x2, y2 = vehicle["bbox"]
            print(
                f"{index}. Típus: {vehicle['label']} | "
                f"Bizonyosság: {vehicle['confidence']:.1%} | "
                f"Koordináták: ({x1}, {y1}) - ({x2}, {y2})"
            )

    except (FileNotFoundError, ValueError) as error:
        print(f"Hiba: {error}")

    except Exception as error:
        print(
            f"Hiba történt a járműfelismerés során: {error}"
        )


if __name__ == "__main__":
    main()
