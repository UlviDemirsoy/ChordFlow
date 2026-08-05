# ChordFlow

MediaPipe Hand Landmarker ile gerçek zamanlı el takibi ve landmark dataset toplama uygulaması.

## Çalıştırma

```bash
npm install
npm run dev
```

- Ana uygulama: `http://localhost:5173/`
- Sol el dataset aracı: `http://localhost:5173/collect`
- Sağ el dataset aracı: `http://localhost:5173/collect/right`
- Sol el realtime classifier testi: `http://localhost:5173/classify`
- Sağ el realtime classifier testi: `http://localhost:5173/classify/right`
- İki el gesture synth: `http://localhost:5173/play`

### Docker (prod)

```bash
docker compose up --build
# veya
docker build -t chordflow . && docker run --rm -p 8080:80 chordflow
```

Açık adres: `http://localhost:8080/play`

Branch’ler: `main` = prod, `develop` = dev. Dataset fotoğrafları ve `landmarks.csv` git’e dahil edilmez; final ONNX modelleri `public/models/` altındadır.

## Dataset toplama

1. `/collect` sayfasını aç.
2. Kamerayı başlat ve sol elin algılandığını kontrol et.
3. Dropdown veya klavyedeki `0–7` tuşlarıyla sınıfı seç (`0 = Unknown`).
4. `Space` ile tek örnek ya da toplu kayıt düğmesiyle 10, 25 veya 50 örnek kaydet.

Her örnek aynı UUID ile ilişkilendirilir:

```text
model/data/left/
├── 0/{uuid}.jpg
├── 1/{uuid}.jpg
├── 2/{uuid}.jpg
├── ...
├── 6/{uuid}.jpg
├── 7/{uuid}.jpg
└── landmarks.csv
```

`landmarks.csv`; UUID, sınıf, fotoğraf yolu, kayıt zamanı, handedness skoru ve 21 landmark için ham `x`, `y`, `z` koordinatlarını içerir. Sınıf `0`, modelin explicit `Unknown` etiketidir.

Dataset yazma API'si yalnızca yerel Vite geliştirme sunucusunda çalışır.

Sağ el collector aynı kayıt akışını `Unknown (0) + Class 1–5` için kullanır:

```text
model/data/right/
├── 0/{uuid}.jpg
├── 1/{uuid}.jpg
├── ...
├── 5/{uuid}.jpg
└── landmarks.csv
```

Sağ el sınıfları akor adlarına bağlı değildir; uygulama katmanı daha sonra
`Class 1–5` çıktılarını istenen akor tiplerine dinamik olarak eşler.

## Sol el modelini eğitme

Jupyter ile adım adım çalıştırmak için:

```bash
jupyter lab model/left_hand_model_training.ipynb
```

Notebook hücrelerini yukarıdan aşağıya çalıştır.

Komut satırından otomatik eğitim için:

```bash
pip install -r model/requirements-training.txt
python model/left_hand_model_training.py
```

Script; landmarkları bileğe göre merkezler, avuç boyutuyla ölçekler, el
rotasyonunu normalize eder ve yalnızca train verisinden hesaplanan mean/std ile
standardize eder. Ardışık kamera karelerinin splitler arasında sızmasını
azaltmak için temporal blok tabanlı train/validation/test ayrımı kullanır.

Her eğitim çalıştırması ayrı bir klasöre kaydedilir:

```text
model/training_outputs/left_hand/run_<timestamp>/
├── left_hand_model.pt
├── left_hand_model_torchscript.pt
├── preprocessing.npz
├── metrics.json
├── classification_report.csv
├── confidence_thresholds.csv
├── training_history.png
├── confusion_matrix.png
└── confidence_analysis.png
```

Sağ el modeli aynı pipeline ile `Unknown (0) + Class 1–5` olarak eğitilir:

```bash
jupyter lab model/right_hand_model_training.ipynb
# veya
python model/right_hand_model_training.py
```

Çıktılar `model/training_outputs/right_hand/run_<timestamp>/` altında tutulur.

## Realtime ONNX classifier

En güncel notebook/script checkpointini tarayıcı modeline dönüştür:

```bash
python model/export_left_hand_onnx.py
python model/verify_left_hand_onnx.py
```

Sağ el modelini export ve verify etmek için:

```bash
python model/export_left_hand_onnx.py --hand right
python model/verify_left_hand_onnx.py --hand right
```

Ardından Vite sunucusu açıkken `http://localhost:5173/classify` adresine git.
Sayfa MediaPipe sol el landmarklarını ONNX Runtime Web'e verir ve Unknown +
1–7 sınıflarının olasılıklarını canlı gösterir. Class `0` explicit Unknown
sonucudur; confidence `%85` veya ilk iki sınıf arasındaki margin `%20` altında
kaldığında sonuç ayrıca `kararsız` olarak reddedilir.

Sağ el testi `/classify/right` adresindedir. Sağ el argmax sınıfı confidence
reddi olmadan son 3 frame içindeki `2/3` çoğunluk filtresine girer.

## Gesture Synth

`/play` tek kamera akışını iki ONNX classifier'a verir. Sol el `1–7`, seçilen
tonalitenin akor kök derecesini belirler. Sağ el sınıfları:

```text
1 → Major
2 → Sus4
3 → Dominant 7
4 → Minor
5 → Diminished
```

Frontend'den 12 tonal kökten biri ve `Major/Minor` gam seçilir. Sayfa açılınca
kamera ve ses otomatik başlar; her iki elin sonucu temporal `2/3` çoğunluk
filtresinden geçtiğinde Web Audio synth akoru çalar.
Unknown, kararsız sonuç, el kaybı, kamera durması veya sekmenin gizlenmesi sesi
yumuşak release ile kapatır.

Sağ el wrist `Y` koordinatı classifier'dan bağımsız olarak expression volume
kontrol eder: `Y=0` maksimum, `Y=1` minimum sestir. EMA smoothing ve deadband
küçük landmark titreşimlerinin ses seviyesini bozmasını engeller.
