
from pathlib import Path
import math

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

# A járműkeretek oldalankénti csökkentésének aránya
BOX_SHRINK_RATIO = 0.20

# A minimális rés a két jármű átlagos méretéhez képest
PARKING_GAP_RATIO = 0.50

# A függőleges eltérés megengedett aránya
MAX_VERTICAL_OFFSET_RATIO = 1.50

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
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))

    return x1, y1, x2, y2


#A járművek közötti rések és férőhelyek meghatározása
def find_parking_spaces(vehicles):
    # A járműveket vízszintes középpont szerint rendezzük
    sorted_vehicles = sorted(
        vehicles,
        key=lambda vehicle: (
            vehicle["bbox"][0] + vehicle["bbox"][2] #100 | közép | 100 --> /2
        ) / 2
    )

    # A feltételezett szabad helyek
    parking_spaces = []

    # A vizsgált rések adatai
    gap_checks = []

    # Megvizsgáljuk a vízszintesen szomszédos járműveket
    for index in range(len(sorted_vehicles) - 1):
        left_vehicle = sorted_vehicles[index]
        right_vehicle = sorted_vehicles[index + 1]

        # A két jármű keretének koordinátái
        lx1, ly1, lx2, ly2 = left_vehicle["bbox"]
        rx1, ry1, rx2, ry2 = right_vehicle["bbox"]

        # A járművek keretének szélessége és magassága
        left_width = lx2 - lx1
        left_height = ly2 - ly1
        right_width = rx2 - rx1
        right_height = ry2 - ry1

        # A két keret közötti vízszintes távolság
        gap = rx1 - lx2

        # A két jármű hosszabbik oldala
        left_size = max(left_width, left_height)
        right_size = max(right_width, right_height)

        # A két jármű átlagos mérete
        reference_size = (left_size + right_size) / 2

        # A minimális rés kiszámítása
        min_gap = reference_size * PARKING_GAP_RATIO

        #!!!!!!!!!Egy becsült parkolóhely szükséges szélessége
        estimated_slot_width = min_gap * 2.45

        # A járművek függőleges középpontja
        left_center_y = (ly1 + ly2) / 2
        right_center_y = (ry1 + ry2) / 2

        # A középpontok függőleges eltérése
        vertical_offset = abs(left_center_y - right_center_y)

        # A nagyobb keret magassága
        reference_height = max(left_height, right_height)

        # A megengedett függőleges eltérés
        max_vertical_offset = (
            reference_height * MAX_VERTICAL_OFFSET_RATIO
        )

        # A két keret függőleges átfedése
        vertical_overlap = min(ly2, ry2) - max(ly1, ry1)

        # Ellenőrizzük, hogy a két jármű nagyjából egy sorban van-e
        same_row = (
            vertical_overlap > 0
            or vertical_offset <= max_vertical_offset
        )

        # Ellenőrizzük, hogy a rés elég széles-e egy helyhez
        enough_gap = gap >= estimated_slot_width

        #Becsüljük meg, hány hely férhet el a résben
        if enough_gap and same_row:
            slot_count = max(
                1,
                math.floor(gap / estimated_slot_width)
            )
        else:
            slot_count = 0

        # A rés középpontja
        center_x = (lx2 + rx1) / 2
        center_y = (
            left_center_y + right_center_y
        ) / 2

        # Eltároljuk a vizsgálat adatait
        check = {
            "left_edge": (lx2, round(left_center_y)),
            "right_edge": (rx1, round(right_center_y)),
            "gap": gap,
            "min_gap": min_gap,
            "slot_width": estimated_slot_width,
            "slot_count": slot_count,
            "reference_size": reference_size,
            "vertical_offset": vertical_offset,
            "max_vertical_offset": max_vertical_offset,
            "enough_gap": enough_gap,
            "same_row": same_row,
            "accepted": slot_count > 0,
            "center": (round(center_x), round(center_y)),
        }

        gap_checks.append(check)

        #Minden becsült férőhelyhez külön középpontot rendelünk
        if slot_count > 0:
            for slot_index in range(slot_count):
                slot_center_x = (
                    lx2
                    + (slot_index + 0.5) * gap / slot_count
                )

                parking_spaces.append({
                    "center": (
                        round(slot_center_x),
                        round(center_y)
                    ),
                    "width": gap / slot_count,
                    "slot_number": slot_index + 1,
                    "total_slots": slot_count,
                })

    return parking_spaces, gap_checks


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

    # Megkeressük a lehetséges szabad helyeket
    parking_spaces, gap_checks = find_parking_spaces(
        detected_vehicles
    )

    # Betöltjük az eredeti képet RGB formátumban
    with Image.open(image_path) as image:
        annotated_image = image.convert("RGB")

    # Létrehozzuk a rajzoló objektumot
    draw = ImageDraw.Draw(annotated_image)

    # A betűméretet a kép szélességéhez igazítjuk
    font_size = max(16, annotated_image.width // 50)

    #Betűtípus betöltése
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Kirajzoljuk a járművek csökkentett kereteit
    for vehicle in detected_vehicles:
        x1, y1, x2, y2 = vehicle["bbox"]

        #A jármű címkéje
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

        # Háttér a címke mögött
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

    # Kirajzoljuk az összes vizsgált rést
    for check in gap_checks:
        # A két keret vizsgált széle
        left_x, left_y = check["left_edge"]
        right_x, right_y = check["right_edge"]

        # Zöld vonal jelzi az elfogadott, piros az elutasított rést
        line_color = (
            (0, 255, 0)
            if check["accepted"]
            else (255, 0, 0)
        )

        # A két vizsgált szélt összekötő vonal
        draw.line(
            (left_x, left_y, right_x, right_y),
            fill=line_color,
            width=4,
        )

        # Függőleges jelölések a boxok szélén
        marker_size = 12

        draw.line(
            (
                left_x, left_y - marker_size,
                left_x, left_y + marker_size,
            ),
            fill=(255, 255, 0),
            width=3,
        )

        draw.line(
            (
                right_x, right_y - marker_size,
                right_x, right_y + marker_size,
            ),
            fill=(255, 255, 0),
            width=3,
        )

        #A rés szélességét és a becsült férőhelyek számát is kiírjuk
        info_text = (
            f"res={check['gap']:.0f}px "
            f"min={check['min_gap']:.0f}px "
            f"helyek={check['slot_count']}"
        )

        # A felirat középpontja
        text_x = round((left_x + right_x) / 2)
        text_y = round(
            (left_y + right_y) / 2
        ) - font_size - 8

        # A felirat mérete
        info_bbox = draw.textbbox((0, 0), info_text, font=font)
        info_width = info_bbox[2] - info_bbox[0]
        info_height = info_bbox[3] - info_bbox[1]

        # A feliratot a képen belül tartjuk
        text_y = max(0, text_y)

        # Háttér a távolság feliratához
        draw.rectangle(
            (
                text_x - info_width // 2 - 4,
                text_y,
                text_x + info_width // 2 + 4,
                text_y + info_height + 6,
            ),
            fill=(0, 0, 0),
        )

        # A távolság felirata
        draw.text(
            (
                text_x - info_width // 2,
                text_y + 2,
            ),
            info_text,
            fill=line_color,
            font=font,
        )

        # A vizsgálat eredménye a konzolra
        print(
            f"Rés: {check['gap']:.1f}px | "
            f"Minimális küszöb: {check['min_gap']:.1f}px | "
            f"Becsült helyszélesség: {check['slot_width']:.1f}px | "
            f"Becsült férőhelyek: {check['slot_count']} | "
            f"Függőleges eltérés: "
            f"{check['vertical_offset']:.1f}px | "
            f"Max.: {check['max_vertical_offset']:.1f}px | "
            f"Egy sorban: {check['same_row']} | "
            f"Elfogadva: {check['accepted']}"
        )

    #Minden becsült férőhely közepére külön felirat kerül
    for space in parking_spaces:
        center_x, center_y = space["center"]

        # A parkolóhely felirata
        text = "1 szabad hely"

        # A felirat mérete
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Középre igazítjuk a feliratot
        text_x = round(center_x - text_width / 2)
        text_y = round(center_y - text_height / 2)

        # Háttér margója
        padding = 6

        # Kék háttér a felirat mögött
        draw.rectangle(
            (
                text_x - padding,
                text_y - padding,
                text_x + text_width + padding,
                text_y + text_height + padding,
            ),
            fill=(0, 100, 255),
        )

        # A szabad hely felirata
        draw.text(
            (text_x, text_y),
            text,
            fill=(255, 255, 255),
            font=font,
        )

    # Az eredménykép mentése
    annotated_image.save(
        OUTPUT_PATH,
        format="JPEG",
        quality=95,
    )

    # Eredmények kiírása
    print(f"\nEredménykép elmentve: {OUTPUT_PATH}")
    print(f"Felismert járművek: {len(detected_vehicles)} db")
    print(f"Becsült szabad parkolóhelyek: {len(parking_spaces)} db")

    return detected_vehicles


def main():
    print("PARKOLÓHELY FELISMERÉS")
    print("-" * 35)

    try:
        # Ellenőrizzük a bemeneti képet
        image_info = load_image(IMAGE_PATH)

        # Kiírjuk a kép adatait
        print("A kép sikeresen betöltve.")
        print(f"Formátum: {image_info['format']}")
        print(
            f"Felbontás: {image_info['size'][0]} x "
            f"{image_info['size'][1]} pixel"
        )
        print(f"Színmód: {image_info['mode']}")
        print("-" * 35)

        # Elindítjuk a felismerést
        print("Járművek keresése a képen...")
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
