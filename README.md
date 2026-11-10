# YOLOv26 Segmentierungsprogramm (SEG_train_custom_data.py)

Dokumentation für das **YOLOv26 Instance Segmentation Training** mit benutzerdefinierten Daten.

---

## Inhalt

1. [Überblick](#überblick)
2. [Systemanforderungen](#systemanforderungen)
3. [Installation](#installation)
4. [Ordnerstruktur](#ordnerstruktur)
5. [Datenaufbereitung](#datenaufbereitung)
6. [Training durchführen](#training-durchführen)
7. [Verwendung des trainierten Modells](#verwendung-des-trainierten-modells)
8. [Hilfreiche Dateien im Projekt](#hilfreiche-dateien-im-projekt)
9. [Häufig gestellte Fragen (FAQ)](#häufig-gestellte-fragen)
10. [Fehlerbehandlung](#fehlerbehandlung)

---

## Überblick

Dieses Projekt basiert auf **YOLOv26** und spezialisiert sich auf **Instance Segmentation** – die Erkennung und pixelgenaue Segmentierung von Objekten in Bildern.

### Kernfunktionalitäten:
- **Training** mit benutzerdefinierten Datensätzen
- **Live-Demo** mit Webcam und Echtzeit-Segmentierung
- **Label-Konvertierung** von LabelMe (JSON) zu YOLO-Format

---

## Systemanforderungen

### Hardware
- **GPU**: NVIDIA GPU mit CUDA-Unterstützung (empfohlen)
- **RAM**: Mindestens 8 GB (16 GB+ empfohlen)
- **Speicher**: Mindestens 10 GB freier Speicherplatz

### Software
- **Python**: 3.8 - 3.11 (3.10 empfohlen)
- **Operating System**: Windows / Linux / macOS

---

## Installation

### 1. Python Environment erstellen (mit venv)

```bash
# Python Virtual Environment erstellen
python -m venv yolov26

# Aktivieren (Windows)
yolov26\Scripts\activate

# Aktivieren (Linux/macOS)
source yolov26/bin/activate
```

### 2. Abhängigkeiten installieren

Alle notwendigen Pakete sind in `requirements.txt` dokumentiert. Installiere sie mit:

```bash
# Im aktivierten Environment
pip install -r requirements.txt

# Oder einzeln für manuelle Installation:
pip install ultralytics==8.4.47
pip install torch==2.11.0 torchvision==0.26.0
pip install opencv-python==4.13.0.92
pip install albumentations==2.0.8
pip install labelme==6.2.0
```

### 3. Überprüfung der Installation

```bash
# Python Version prüfen
python --version

# Ultralytics Version überprüfen
python -c "from ultralytics import YOLO; print(YOLO.__version__)"

# PyTorch überprüfen
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA verfügbar: {torch.cuda.is_available()}')"
```

---

## Ordnerstruktur

Die folgende Ordnerstruktur **MUSS** eingehalten werden:

```
YOLOv26/
│
├── datasets/                          # Datensatz-Ordner
│   ├── box-seg/                       # Beispiel-Datensatz 1
│   │   ├── box-seg.yaml              # Datensatz-Konfiguration
│   │   ├── images/                   # Bilddateien
│   │   │   ├── train/               # Trainingsdaten
│   │   │   └── val/                 # Validierungsdaten
│   │   ├── json_labels/              # LabelMe-Annotationen (JSON)
│   │   └── labels/                   # YOLO-Format Labels
│   │       ├── train/               # Trainingslabels (.txt)
│   │       └── val/                 # Validierungslabels (.txt)
│   │
│   ├── box_zylinder-seg/             # Beispiel-Datensatz 2
│   │   ├── box_zylinder-seg.yaml
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── json_labels/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   │
│   ├── coco8-seg/                    # COCO8 Beispiel-Datensatz
│   │   ├── coco8-seg.yaml
│   │   ├── images/
│   │   └── labels/
│   │
│   └── [IHR_NEUER_DATENSATZ]/        # Ihr neuer Datensatz
│       ├── [DATENSATZNAME].yaml      # (siehe unten)
│       ├── images/
│       │   ├── train/                # (70% der Bilder)
│       │   └── val/                  # (30% der Bilder)
│       ├── json_labels/              # Optional: Originale LabelMe-Dateien
│       └── labels/
│           ├── train/                # (.txt Dateien, eine pro Bild)
│           └── val/
│
├── runs/                              # Ausgabe-Ordner für Trainingsergebnisse
│   └── segment/
│       ├── train/                     # Training-Run 1
│       │   ├── weights/
│       │   │   ├── best.pt           # Bestes Modell
│       │   │   └── last.pt           # Letztes Checkpoint
│       │   └── results.csv
│       ├── train-2/                   # Training-Run 2
│       └── ...
│
├── results/                           # Ordner für Visualisierungen
│   ├── SEG_pca/                       # PCA-Plots der Segmentierung
│   ├── SEG_snapshots/                 # Screenshots aus Live-Demo
│   ├── OD_pca/                        # Detection PCA
│   └── OD_snapshots/                  # Detection Screenshots
│
├── yolov26/                           # Python Virtual Environment
│
├── SEG_train_custom_data.py          # ⭐ HAUPTDATEI: Trainiert mit benutzerdefinierten Daten
├── SEG_train_coco8.py                # Training mit COCO8 Beispieldatensatz
├── SEG_demo_cv2.py                   # Live-Demo mit Webcam
├── SEG_json2yolo.py                  # Konvertierung: JSON → YOLO-Labels
├── SEG_object.py                     # Klasse für Segmentierungs-Objekte
│
├── OD_*.py                            # Object Detection Dateien (nicht primärer Fokus)
├── function_pool.py                   # Hilfsfunktionen
├── pixel2robot.py                     # Pixel zu Robot-Koordinaten Konvertierung
│
├── requirements.txt                   # Abhängigkeiten
├── yolo26n-seg.pt                     # ⭐ Vortrainiertes YOLOv26n Segmentierungsmodell
├── yolo26n.pt                         # Vortrainiertes YOLOv26n Detection Modell
│
├── robot_poses.json                   # Robot-Positionen-Datei
├── output_c2f.json                    # Kamerakalibrierung
└── output_wp2camera.json              # Worldpoint zu Camera Kalibrierung
```

### Wichtige Anmerkungen zur Ordnerstruktur:

1. **Bilderordner**: Müssen in `images/train/` und `images/val/` unterteilt sein
2. **Label-Ordner**: Entsprechende `.txt` Dateien in `labels/train/` und `labels/val/`
3. **Datensatz-YAML**: Muss auf die relativen Pfade zeigen
4. **Pfad-Trennzeichen**: Unter Windows: `\\` oder `/` (beide funktionieren in YAML)

---

## Datenaufbereitung

### Schritt 1: Bilder sammeln und annotieren

#### Mit LabelMe
```bash
# LabelMe starten
labelme

# Bilder laden → Polygone zeichnen → Speichern (JSON-Format)

### Schritt 2: JSON Labels zu YOLO-Format konvertieren

Verwenden Sie `SEG_json2yolo.py` um die LabelMe JSON-Dateien in das YOLO-Format zu konvertieren:

```bash
# Editieren Sie die Datei und setzen Sie die Pfade:
# json_dir = Path("datasets\\mein_datensatz\\json_labels")
# output_dir = Path("datasets\\mein_datensatz\\labels\\train")
# classes = {"klasse1": 0, "klasse2": 1, ...}

python SEG_json2yolo.py
```

**Ausgabe**: `.txt` Dateien im YOLO-Segmentierungsformat:
```
# Beispiel: 0 0.12 0.15 0.45 0.32 0.78 0.92
# Format: class_id x1_norm y1_norm x2_norm y2_norm ... xn_norm yn_norm
```

**Wichtig**: Die generierten `.txt` Dateien müssen manuell in `datasets/mein_datensatz/labels/train/` verschoben werden.

### Schritt 3: Bilder und Labels in Train/Val Split aufteilen

Nach der Label-Konvertierung verwenden Sie das automatisierte Tool `split_dataset.py`:

```bash
# Environment aktivieren
yolov26\Scripts\activate

# Programm starten
python split_dataset.py

# Folgende Eingabe erwartet:
# Datensatz-Name eingeben (z.B. 'box_zylinder-seg'): mein_datensatz
# Fortfahren? (j/n): j
```

**Was das Programm macht:**
-  Durchsucht `images/train` und `labels/train`
-  Wählt zufällig 30% der Bilder aus
-  Verschiebt diese Bilder + entsprechende Labels in `images/val` und `labels/val`
-  70% verbleiben in `train`
-  Überprüft, dass jedes Bild ein entsprechendes Label hat

### Schritt 4: YAML-Konfigurationsdatei erstellen

Erstellen Sie eine `.yaml` Datei für Ihren Datensatz (z.B. `mein_datensatz.yaml`):

```yaml
# Datensatz-Konfiguration für YOLO Segmentierung
path: datasets/mein_datensatz              # Pfad zum Datensatz-Ordner
train: images/train                        # Relativer Pfad zu Trainingsbilder
val: images/val                            # Relativer Pfad zu Validierungsbilder

nc: 2                                       # Anzahl der Klassen
names: ['klasse1', 'klasse2']              # Klassennamen (Index = Class ID)
```

### Beispiel-Struktur nach Aufbereitung:

```
datasets/mein_datensatz/
├── mein_datensatz.yaml
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ... (70% der Bilder)
│   └── val/
│       ├── img_071.jpg
│       └── ... (30% der Bilder)
└── labels/
    ├── train/
    │   ├── img_001.txt
    │   ├── img_002.txt
    │   └── ...
    └── val/
        ├── img_071.txt
        └── ...
```

---

## Training durchführen

### Schritt 1: SEG_train_custom_data.py konfigurieren

Öffnen Sie die Datei und passen Sie diese Parameter an:

```python
# Linie 8-10: Modell laden (Option A oder B)
model = YOLO("yolo26n-seg.pt")  # Option A: Mit vortrainiertem Modell (EMPFOHLEN)
# model = YOLO("yolo26n-seg.yaml")  # Option B: Von Grund auf neu (weniger empfohlen)

# Linie 14-15: Training starten
results = model.train(
    data="datasets/MEIN_DATENSATZ/mein_datensatz.yaml",  # ANPASSEN!
    epochs=100,              # Anzahl der Epochen (50-100 üblich)
    imgsz=640,              # Bildgröße (640 ist Standard)
    rect=True,              # Rechteckige Bilder (schneller Training)
    batch=16                # Batch-Größe (abhängig von GPU-Speicher)
)
```

### Schritt 2: Batch-Größe anpassen (abhängig von GPU-Speicher)

```python
# GPU-Speicher vs. Batch-Size
# RTX 4090 / RTX 4080: batch=32 oder batch=64
# RTX 3080 / RTX 4070:  batch=16 oder batch=32
# RTX 2080 / RTX 3070:  batch=8 oder batch=16
# CPU-only:            batch=4 oder batch=8 (sehr langsam!)
```

### Schritt 3: Training starten

```bash
# Environment aktivieren (falls nicht aktiv)
yolov26\Scripts\activate  # Windows

# Training starten
python SEG_train_custom_data.py
```

### Training-Ausgaben

Nach erfolgreichem Training finden Sie die Ergebnisse in:

```
runs/segment/train/
├── weights/
│   ├── best.pt          # ⭐ Bestes Modell (verwenden Sie dies!)
│   └── last.pt          # Letzter Checkpoint
├── results.csv          # Trainings-Metriken
├── confusion_matrix.png # Konfusions-Matrix
├── F1_curve.png         # F1-Kurve
├── PR_curve.png         # Precision-Recall Kurve
└── results.png          # Übersichts-Plot
```

---

## Verwendung des trainierten Modells

### Option 1: Live-Demo mit Webcam (SEG_demo_cv2.py)

```python
# Bearbeiten Sie SEG_demo_cv2.py:
MODEL_PATH = "runs/segment/train/weights/best.pt"  #  Anpassen!

# Starten Sie dann:
python SEG_demo_cv2.py

# Bedienung: Drücken Sie 'q' zum Beenden
```

### Option 2: Bilder in Batch verarbeiten

```python
from ultralytics import YOLO

# Modell laden
model = YOLO("runs/segment/train/weights/best.pt")

# Ordner mit Bildern
results = model.predict(
    source="path/to/images",
    conf=0.5,           # Confidence Threshold
    save=True,          # Speichern Sie die Ergebnisse
    save_txt=True       # Speichern Sie die Labels
)
```

## Hilfreiche Dateien im Projekt

### Hauptdateien (Segmentierung)

| Datei | Beschreibung |
|-------|-------------|
| **SEG_train_custom_data.py** | ⭐ Hauptdatei zum Training mit benutzerdefinierten Daten |
| SEG_train_coco8.py | Training mit vordefiniertem COCO8-Datensatz (Schnelltest) |
| SEG_demo_cv2.py | Live-Demo mit Webcam und Echtzeit-Segmentierung |
| SEG_json2yolo.py | Konvertiert LabelMe JSON-Dateien zu YOLO-TXT-Format |
| SEG_object.py | Klasse für erweiterte Segmentierungsanalyse (3D-Rekonstruktion) |
| **split_dataset.py** | Teilt Trainings-Daten automatisch in Train/Val (70/30) auf |

### Nebenfiles (Erkennung)

| Datei | Beschreibung |
|-------|-------------|
| OD_json2yolo.py | JSON zu YOLO-Konvertierung für Detection |

### Utility-Dateien

| Datei | Beschreibung |
|-------|-------------|
| function_pool.py | Sammlung von Hilfsfunktionen |
| pixel2robot.py | Konvertierung Pixel-Koordinaten zu Robot-Koordinaten |
| pick_place.py | Anwendung für Roboter Pick & Place |

---

## Fehlerbehandlung

### Fehler: "FileNotFoundError: datasets/..."

**Grund**: YAML-Pfade falsch oder Ordner nicht vorhanden

**Lösung**:
```python
# Überprüfen Sie die YAML-Datei:
# 1. Pfade sind relativ zum Projektordner
# 2. Ordner "images/train" existiert wirklich
# 3. Dateien in den Ordnern vorhanden sind

### Fehler: "Expected x, got y" bei Label-Daten

**Grund**: Label-Format stimmt nicht mit YOLO überein

**Lösung**:
```bash
# Überprüfen Sie eine .txt-Datei aus labels/train:
# Format muss sein: class_id x1 y1 x2 y2 ... xn yn
# Alle Werte zwischen 0 und 1 normalisiert

# Beispiel richtig:
# 0 0.123 0.456 0.234 0.567 0.345 0.678

# Beispiel falsch:
# 0 123 456 234 567  (nicht normalisiert!)
```

### Fehler: "ModuleNotFoundError: No module named 'ultralytics'"

**Grund**: Abhängigkeiten nicht installiert oder Environment nicht aktiviert

**Lösung**:
```bash
# Environment aktivieren:
yolov26\Scripts\activate  # Windows

# Abhängigkeiten erneut installieren:
pip install -r requirements.txt

# Oder einzeln:
pip install ultralytics
```

### Fehler: "Webcam nicht erkannt" in SEG_demo_cv2.py

**Grund**: CAMERA_INDEX falsch oder Webcam blockiert

**Lösung**:
```python
# In SEG_demo_cv2.py:
CAMERA_INDEX = 0  # Intern-Webcam
# oder
CAMERA_INDEX = 1  # Externe Webcam (je nach System)

# Oder CAMERA_BACKEND ändern:
CAMERA_BACKEND = cv2.CAP_MSMF  # Statt cv2.CAP_DSHOW
```
