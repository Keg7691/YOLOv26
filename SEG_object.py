import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.decomposition import PCA
import os
from datetime import datetime
import json

# --- Konfiguration ---
CAMERA_INDEX   = 0                     # 0 = intern, 1 = extern
CAMERA_BACKEND = cv2.CAP_DSHOW          # cv2.CAP_DSHOW | cv2.CAP_MSMF | cv2.CAP_ANY
WINDOW_NAME    = "SEG Object Demo"
MODEL_PATH     = "runs/segment/train-3/weights/best.pt"        # Pfad zum trainierten Segmentierungsmodell
CALIB_PATH     = "output_wp2camera.json"       # Pfad zur Kamerakalibrierung
OUTPUT_DIR     = "results/SEG_pca"             # Ordner für PCA-Plots
SNAPSHOT_DIR   = "results/SEG_snapshots"       # Ordner für Kameraaufnahmen
USE_FULL_MASK  = False                   # True = alle Maskenpixel | False = nur Randpixel
POINT_COLOR    = "blue"                 # Farbe der Punktwolke
PCA_ARROW_SCALE = 60                    # Skalierung der PCA-Achsenpfeile (in Pixeln)
OBJECT_HEIGHT  = 30                    # Höhe des Werkstücks über Referenzebene (in mm)

# Output-Ordner automatisch erstellen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def load_camera_calibration(filepath):
    """Kamera-Matrix und Verzerrungskoeffizienten aus JSON laden."""
    with open(filepath, 'r') as f:
        calib_data = json.load(f)
    
    camera_matrix = np.array(calib_data['camera_matrix'], dtype=float)
    dist_coeffs = np.array(calib_data['dist_coefs'], dtype=float).flatten()
    
    print(f"[INFO] Kamerakalibrierung geladen:")
    print(f"       fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
    print(f"       cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
    
    return camera_matrix, dist_coeffs


# Modell laden
model = YOLO(MODEL_PATH)

# Kamerakalibrierung laden
camera_matrix, dist_coeffs = load_camera_calibration(CALIB_PATH)

# Kamera öffnen
cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Fenster einmalig erstellen (bleibt dauerhaft offen)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

def get_point_cloud(mask, use_full_mask=True):
    """Punktwolke aus einer binären Segmentierungsmaske extrahieren."""
    if use_full_mask:
        # Alle Pixel innerhalb der Maske (y, x) -> umkehren zu (x, y)
        yx = np.column_stack(np.where(mask > 0))
        return yx[:, ::-1].astype(float)  # (x, y)
    else:
        # Nur Randpixel (Kontur)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return np.array([])
        return np.vstack([c.reshape(-1, 2) for c in contours]).astype(float)


def rotation_matrix_to_euler_angles(R):
    """Rotationsmatrix zu [Roll(Z), Pitch(Y), Yaw(X)] in Grad."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    if not singular:
        yaw_x = np.arctan2(R[2, 1], R[2, 2])
        pitch_y = np.arctan2(-R[2, 0], sy)
        roll_z = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw_x = np.arctan2(-R[1, 2], R[1, 1])
        pitch_y = np.arctan2(-R[2, 0], sy)
        roll_z = 0
    
    return np.degrees([roll_z, pitch_y, yaw_x])


def compute_pca(points, camera_matrix, dist_coeffs, object_height=0):
    """
    PCA auf 2D-Punktwolke anwenden und 3D-Orientierung berechnen.
    
    Args:
        points: 2D-Bildpunkte (N x 2)
        camera_matrix: Intrinsische Kamera-Matrix
        dist_coeffs: Verzerrungskoeffizienten
        object_height: Höhe des Werkstücks über der Referenzebene (in mm)
    
    Returns:
        mean, components, variances, angle_2d, euler_angles, rotation_matrix, tvec
    """
    # Standard 2D-PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    print(f"[INFO] PCA-Eigenwerte: {pca.singular_values_}")
    
    # 2D-Orientierungswinkel (im Bildraum)
    # Principal axes in feature space, representing the directions of maximum variance in the data.
    main_axis = pca.components_[0]
    angle_2d = np.degrees(np.arctan2(main_axis[1], main_axis[0]))
    
    # Für 3D-Pose: Annahme dass Werkstück flach liegt (z=0 Ebene)
    # Wir nutzen die Eckpunkte der minimierten Bounding Box
    hull = cv2.convexHull(points.astype(np.float32))
    rect = cv2.minAreaRect(hull)
    box_points = cv2.boxPoints(rect).astype(np.float32)
    
    # 3D-Objektpunkte (Rechteck mit realen Dimensionen aus minAreaRect)
    width, height = rect[1]
    object_points_3d = np.array([
        [-width/2, -height/2, object_height],
        [width/2, -height/2, object_height],
        [width/2, height/2, object_height],
        [-width/2, height/2, object_height]
    ], dtype=np.float32)
    
    # solvePnP für 3D-Pose
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points_3d,
            box_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if success:
            # Rotationsvektor zu Matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
        else:
            rotation_matrix = None
            euler_angles = None
            tvec = None
    except:
        rotation_matrix = None
        euler_angles = None
        tvec = None
    
    return pca.mean_, pca.components_, pca.explained_variance_, angle_2d, euler_angles, rotation_matrix, tvec


def save_plot(all_points, all_centers, all_components, all_variances, timestamp, all_angles_2d=None, all_euler_angles=None):
    """Punktwolken und PCA-Achsen aller Objekte in einem Plot speichern."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = [POINT_COLOR, "green", "cyan", "orange", "purple"]

    for i, (points, center, components, variances) in enumerate(
        zip(all_points, all_centers, all_components, all_variances)
    ):
        color = colors[i % len(colors)]
        
        # Label mit Winkel-Informationen erweitern
        if all_angles_2d and i < len(all_angles_2d):
            label_pts = f"Objekt {i + 1} – Punktwolke (2D: {all_angles_2d[i]:.1f}°)"
        else:
            label_pts = f"Objekt {i + 1} – Punktwolke"
        
        if all_euler_angles and i < len(all_euler_angles) and all_euler_angles[i] is not None:
            euler = all_euler_angles[i]
            label_ctr = f"Objekt {i + 1} – Mittelpunkt (R:{euler[0]:.0f}° P:{euler[1]:.0f}° Y:{euler[2]:.0f}°)"
        else:
            label_ctr = f"Objekt {i + 1} – Mittelpunkt"

        # Punktwolke
        ax.scatter(
            points[:, 0], points[:, 1],
            s=1, color=color, alpha=0.4, label=label_pts
        )

        # Mittelpunkt
        ax.plot(*center, marker="+", markersize=4, color="red", label=label_ctr)
        # ax.plot(*center, marker="+", markersize=14, color="red", label=label_ctr)

        # PCA-Achsen als rote Pfeile (skaliert nach Varianz)
        scale = np.sqrt(variances)
        for comp, s in zip(components, scale):
            ax.annotate(
                "",
                xy=(center[0] + comp[0] * s * PCA_ARROW_SCALE / scale[0],
                    center[1] + comp[1] * s * PCA_ARROW_SCALE / scale[0]),
                xytext=(center[0], center[1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=2)
            )

    ax.set_aspect("equal")
    ax.invert_yaxis()   # Bildkoordinaten: Y zeigt nach unten
    ax.set_title("Punktwolke & PCA-Orientierung")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.legend(loc="upper right", markerscale=5)

    filepath = os.path.join(OUTPUT_DIR, f"pca_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Plot gespeichert: {filepath}")


def save_snapshot(frame, timestamp):
    """Kameraaufnahme (annotiertes Bild) speichern."""
    filepath = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
    cv2.imwrite(filepath, frame)
    print(f"[INFO] Aufnahme gespeichert: {filepath}")


def process_snapshot(frame, annotated_frame, results, camera_matrix, dist_coeffs):
    """Snapshot auslösen: Maske extrahieren, PCA berechnen, 3D-Orientierung ermitteln."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results[0].masks is None or len(results[0].masks) == 0:
        print("[WARN] Kein Objekt erkannt – kein Snapshot gespeichert.")
        return

    all_points    = []
    all_centers   = []
    all_components = []
    all_variances  = []
    all_angles_2d  = []
    all_euler_angles = []

    for idx, mask_tensor in enumerate(results[0].masks.data):
        mask = mask_tensor.cpu().numpy()
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        points = get_point_cloud(mask_resized, use_full_mask=USE_FULL_MASK)

        if len(points) < 10:
            print(f"[WARN] Objekt {idx + 1}: Zu wenige Punkte für PCA – übersprungen.")
            continue

        center, components, variances, angle_2d, euler, R, t = compute_pca(
            points, camera_matrix, dist_coeffs, OBJECT_HEIGHT
        )
        
        print(f"[INFO] Objekt {idx + 1}:")
        print(f"       Position: x={center[0]:.1f}, y={center[1]:.1f}")
        print(f"       2D-Winkel: {angle_2d:.1f}°")
        if euler is not None:
            print(f"       3D-Orientierung (Roll, Pitch, Yaw): {euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°")
            if t is not None:
                print(f"       Translation (x,y,z): {t[0][0]:.1f}, {t[1][0]:.1f}, {t[2][0]:.1f} mm")

        all_points.append(points)
        all_centers.append(center)
        all_components.append(components)
        all_variances.append(variances)
        all_angles_2d.append(angle_2d)
        all_euler_angles.append(euler)

    if all_points:
        save_plot(all_points, all_centers, all_components, all_variances, 
                  timestamp, all_angles_2d, all_euler_angles)
        save_snapshot(annotated_frame, timestamp)
    else:
        print("[WARN] Keine verwertbaren Masken gefunden.")


# --- Hauptschleife ---
print(f"[INFO] Leertaste = Snapshot  |  'q' = Beenden")
print(f"[INFO] Plots werden gespeichert in: {os.path.abspath(OUTPUT_DIR)}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow(WINDOW_NAME, annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):      # Leertaste: Snapshot auslösen
        process_snapshot(frame, annotated_frame, results, camera_matrix, dist_coeffs)
    elif key == ord("q"):    # q: Beenden
        break

cap.release()
cv2.destroyAllWindows()
