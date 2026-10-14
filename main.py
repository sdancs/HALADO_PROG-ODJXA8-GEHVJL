
#kényelmesen tudunk fájl- és mappaútvonalakat kezelni
from pathlib import Path

# A pillow, pil könyvtárból importáljuk az Image osztályt
from PIL import Image, UnidentifiedImageError


# Meghatározzuk a projekt főmappájának elérési útját
# A __file__ az aktuális Python-fájl helyére mutat
PROJECT_DIR = Path(__file__).resolve().parent

# Összeállítjuk a tesztkép teljes elérési útját, projektmappába keresünk
IMAGE_PATH = PROJECT_DIR / "adatok" / "felulnezet.jpg"



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
            #fontos adatokat eltároljuk egy szótárba:
            image_info = {
                "format": image.format,
                "size": image.size,
                "mode": image.mode,
            }

        # Visszaadjuk a kép tulajdonságait tartalmazó szótárat:
        return image_info

    # Kezeljük azokat a hibákat, ha előforudltak:
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"A fájl nem olvasható érvényes képként: {image_path}"
        ) from error


def main():
    print("PARKOLÓHELY FELISMERÉS")
    print("-" * 35)

    # a fent megírt függvénnyel meghívjuk a képet, eltároljuk
    try:
        image_info = load_image(IMAGE_PATH)
        # Jelezzük, hogy a kép betöltése és ellenőrzése sikeres volt, mivel nem volt a függvényben hiba
        print("A kép sikeresen betöltve.")
        print(f"Formátum: {image_info['format']}")
        print(f"Felbontás: {image_info['size'][0]} x {image_info['size'][1]} pixel")
        print(f"Színmód: {image_info['mode']}")

    # Kivétel kezelés, ha a a load_image() függvény ide dobja
    except (FileNotFoundError, ValueError) as error:
        print(f"Hiba: {error}")
if __name__ == "__main__":
    main()
