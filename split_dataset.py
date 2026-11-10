"""
Dataset Split Tool für YOLO Segmentierung
==========================================

Dieses Programm teilt Trainings-Daten in Train/Val Split auf:
- 70% der Bilder und Labels bleiben in 'train'
- 30% der Bilder und Labels werden in 'val' verschoben

Workflow:
1. Alle annotierten Bilder sind in: datasets/[DATENSATZ]/images/train
2. Alle konvertierten YOLO-Labels sind in: datasets/[DATENSATZ]/labels/train
3. Dieses Programm verschiebt automatisch 30% der Bilder+Labels in 'val'

Verwendung:
    python split_dataset.py
"""

import shutil
from pathlib import Path
import random
from datetime import datetime


def validate_dataset_path(base_path):
    """Überprüft, ob alle notwendigen Ordner existieren."""
    images_train = base_path / "images" / "train"
    labels_train = base_path / "labels" / "train"
    
    if not images_train.exists():
        raise FileNotFoundError(f"❌ Ordner nicht gefunden: {images_train}")
    
    if not labels_train.exists():
        raise FileNotFoundError(f"❌ Ordner nicht gefunden: {labels_train}")
    
    return images_train, labels_train


def create_val_directories(base_path):
    """Erstellt val-Ordner, falls sie nicht existieren."""
    images_val = base_path / "images" / "val"
    labels_val = base_path / "labels" / "val"
    
    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)
    
    return images_val, labels_val


def get_image_files(images_train):
    """Sammelt alle Bilddateien (.jpg, .png) – Case-insensitive, dedupliziert."""
    supported_formats = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", 
                        "*.Jpg", "*.Jpeg", "*.Png", "*.jPeg")
    images = []
    seen_paths = set()  # Verhindert Duplikate (Windows: case-insensitive FS)
    
    for fmt in supported_formats:
        for img_path in images_train.glob(fmt):
            # Normalisierter Pfad als String (für case-insensitive Deduplication)
            normalized = str(img_path.resolve()).lower()
            if normalized not in seen_paths:
                seen_paths.add(normalized)
                images.append(img_path)
    
    return sorted(images)


def get_corresponding_label(image_path, labels_train):
    """Findet die entsprechende Label-Datei (.txt) für ein Bild."""
    label_name = image_path.stem + ".txt"
    label_path = labels_train / label_name
    
    if not label_path.exists():
        raise FileNotFoundError(
            f"   Label nicht gefunden für: {image_path.name}\n"
            f"   Erwartet: {label_path}\n"
            f"   → Bild wird übersprungen."
        )
    
    return label_path


def split_dataset(dataset_path, val_split=0.3, random_seed=42):
    """
    Teilt den Datensatz auf und verschiebt Bilder + Labels.
    
    Args:
        dataset_path (str/Path): Pfad zum Datensatz (z.B. 'datasets/mein_datensatz')
        val_split (float): Anteil der Validierungsdaten (0.3 = 30%)
        random_seed (int): Seed für reproduzierbare Splits
    """
    
    # Pfade definieren
    base_path = Path(dataset_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"❌ Datensatz-Ordner nicht gefunden: {base_path}")
    
    print(f"\n{'='*70}")
    print(f" Dataset Split Tool - YOLO Segmentierung")
    print(f"{'='*70}")
    print(f"Datensatz: {base_path.absolute()}")
    print(f"Validierungs-Split: {val_split*100:.0f}%")
    print(f"{'='*70}\n")
    
    # Ordner überprüfen
    print("✓ Überprüfe Ordner-Struktur...")
    try:
        images_train, labels_train = validate_dataset_path(base_path)
        images_val, labels_val = create_val_directories(base_path)
        print(f"  ✓ images/train: {images_train}")
        print(f"  ✓ labels/train: {labels_train}")
        print(f"  ✓ images/val: {images_val}")
        print(f"  ✓ labels/val: {labels_val}\n")
    except FileNotFoundError as e:
        print(f"  {e}")
        return False
    
    # Bilder sammeln
    print("✓ Sammle Bilddateien...")
    images = get_image_files(images_train)
    
    if not images:
        print(f"  ❌ Keine Bilddateien in {images_train} gefunden!")
        return False
    
    print(f"  ✓ {len(images)} Bilddateien gefunden\n")
    
    # Zufällig auswählen (für Reproduzierbarkeit)
    random.seed(random_seed)
    random.shuffle(images)
    
    num_val = int(len(images) * val_split)
    num_train = len(images) - num_val
    
    val_images = images[:num_val]
    
    print(f"Aufteiling:")
    print(f"  • Training: {num_train} Bilder ({(num_train/len(images))*100:.1f}%)")
    print(f"  • Validierung: {num_val} Bilder ({(num_val/len(images))*100:.1f}%)")
    print(f"  • Total: {len(images)} Bilder\n")
    
    # Verschieben
    print(f"Verschiebe Bilder und Labels in Validierungs-Ordner...\n")
    
    moved_count = 0
    skipped_count = 0
    errors = []
    
    for idx, image_path in enumerate(val_images, 1):
        try:
            # Entsprechendes Label finden
            try:
                label_path = get_corresponding_label(image_path, labels_train)
            except FileNotFoundError as e:
                print(f"  ⚠️  ({idx}/{num_val}) {e}")
                skipped_count += 1
                continue
            
            # Bild verschieben
            new_image_path = images_val / image_path.name
            shutil.move(str(image_path), str(new_image_path))
            
            # Label verschieben
            new_label_path = labels_val / label_path.name
            shutil.move(str(label_path), str(new_label_path))
            
            moved_count += 1
            
            if idx % 10 == 0 or idx == num_val:
                print(f"  ✓ ({idx}/{num_val}) Verarbeitet")
        
        except Exception as e:
            error_msg = f"({idx}/{num_val}) Fehler bei {image_path.name}: {str(e)}"
            print(f"  ❌ {error_msg}")
            errors.append(error_msg)
            skipped_count += 1
    
    # Zusammenfassung
    print(f"\n{'='*70}")
    print(f"Split abgeschlossen!")
    print(f"{'='*70}")
    print(f"✓ Erfolgreich verschoben: {moved_count} Bilder + Labels")
    
    if skipped_count > 0:
        print(f" Übersprungen: {skipped_count} Bilder (Label nicht gefunden)")
    
    if errors:
        print(f"\n❌ Fehler aufgetreten:")
        for error in errors:
            print(f"   • {error}")
    
    # Finale Überprüfung
    print(f"\n{'='*70}")
    print(f"Finale Statistik:")
    print(f"{'='*70}")
    
    final_train_images = len(list(images_train.glob("*.jpg"))) + \
                        len(list(images_train.glob("*.png"))) + \
                        len(list(images_train.glob("*.jpeg")))
    final_val_images = len(list(images_val.glob("*.jpg"))) + \
                       len(list(images_val.glob("*.png"))) + \
                       len(list(images_val.glob("*.jpeg")))
    
    final_train_labels = len(list(labels_train.glob("*.txt")))
    final_val_labels = len(list(labels_val.glob("*.txt")))
    
    print(f"\nimages/train: {final_train_images} Bilder")
    print(f"images/val:   {final_val_images} Bilder")
    print(f"\nlabels/train: {final_train_labels} Labels")
    print(f"labels/val:   {final_val_labels} Labels")
    print(f"\nTotal: {final_train_images + final_val_images} Bilder, " \
          f"{final_train_labels + final_val_labels} Labels")
    
    # Warnungen
    if final_train_labels != final_train_images:
        print(f"\nWARNUNG: images/train ({final_train_images}) != labels/train ({final_train_labels})")
    
    if final_val_labels != final_val_images:
        print(f"WARNUNG: images/val ({final_val_images}) != labels/val ({final_val_labels})")
    
    print(f"\n{'='*70}\n")
    
    return moved_count > 0


def main():
    """Hauptfunktion mit Benutzer-Eingabe."""
    
    print("\nYOLO Datensatz Split Tool")
    print("="*70)
    print("Dieses Programm teilt Ihre annotierten Trainingsdaten auf:")
    print("  • 70% Training")
    print("  • 30% Validierung")
    print("\nVoraussetzungen:")
    print("  ✓ Alle Bilder sind in: datasets/[DATENSATZ]/images/train")
    print("  ✓ Alle Labels sind in: datasets/[DATENSATZ]/labels/train")
    print("="*70)
    
    # Benutzer-Eingabe
    while True:
        dataset_name = input("\nDatensatz-Name eingeben (z.B. 'box_zylinder-seg'): ").strip()
        
        if not dataset_name:
            print("❌ Bitte einen Datensatz-Namen eingeben!")
            continue
        
        dataset_path = Path("datasets") / dataset_name
        
        if not dataset_path.exists():
            print(f"❌ Pfad nicht gefunden: {dataset_path}")
            print(f"   Vorhandene Datensätze: {list(Path('datasets').iterdir())}")
            continue
        
        break
    
    # Bestätigung
    print(f"\n✓ Datensatz ausgewählt: {dataset_path}")
    confirm = input("Fortfahren? (j/n): ").strip().lower()
    
    if confirm != 'j':
        print("❌ Abgebrochen.")
        return False
    
    # Split durchführen
    success = split_dataset(str(dataset_path), val_split=0.3)
    
    if success:
        print("✅ Datensatz erfolgreich aufgeteilt!")
        return True
    else:
        print("❌ Fehler beim Splitten des Datensatzes.")
        return False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Abgebrochen durch Benutzer.")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
