Parkolóhely felismerő rendszer
A program főleg párhuzamos parkolás esetén működik optimálisan.

A projekt célja:
Python alapú alkalmazás, amely parkolókról készült képek alapján megpróbálja a foglaltságot meghatározni.

Tervezett funkciók:
-kép betöltés
-járművek felismerése
-szabad helyek meghatározása
-eredmény megjelenítése


Kép forrása:
pexels.com, maps.google.com

Requirementsek telepítése:
ultralytics
Pillow

A jelenlegi program a betöltendő kép ellenőrzésével kezd, hogy létezik-e. Ha ez valid, akkor a következő lépésben egy előre betanított YOLOv8 modellt tölt be, amely segítségével felismeri a képen található járműveket oldalnézet esetén. Ezeket a felismert objektumokat eltároljuk a programban, valamint képre is illesztjük kis összehúzás után. Az összehúzás azért volt indokolt, mert a YOLO modell túl nagy keretet rajzolt, így akartuk elekrülni az igaz hamis adatokat. Ezután ezeket a kereteket egy listában vízszintesen haladva balról jobbra a koordináták alapján rendezésre kerülnek. Ezen kívül a vonal méretének kiszámítása az azonosított járművek méretéből történik (két jármű hosszabik oldalának átlagából számított érték 50%-a, ez az érték még egy konstans értékkel is szorozva van, hogy a vonal eléggé széles legyen), ez alapján biztosított, hogy nagyobb felbontású kép esetén és kis felbontás esetén is optimálisan működjön. Ha a vonal elég széles és vízszintes átfedés is található, akkor ott legalább 1 szabad hely van. Ha a vonal szélesebb, akkor akár 2, vagy több hely is rendelkezésre állhat. Fontos, hogy egyéb akadályokat a szoftver nem ismer fel, csak két autó közti szabad helyeket vizsgál. Az eredményekről példaképek is megtalálhatóak. Párhuzamos parkolás esetén több szabadhely felismerése is relatív nagy pontosággal történik, viszont merőleges parkolás esetén 1 hely meghatározása stabil, mivel a 2.45-ös fix szorzó túlzó itt.