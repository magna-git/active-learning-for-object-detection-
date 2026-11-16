
from pathlib import Path
import csv
import shutil

import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

ROOT = Path.cwd()
DATASET = ROOT / "dataset_experiences" / "kmeans"
RUNS = ROOT / "runs"
YAML = ROOT / "data_kmeans.yaml"

B = 16
SEED = 42


def count_images(folder):
    return len(list(folder.glob("*.jpg")))


def select_images(model, pool_dir):
    images = sorted(pool_dir.glob("*.jpg"))
    features = []

    # 1. Extraire les caractéristiques avec YOLO
    for image in images:
        embedding = model.embed(
            source=str(image),
            imgsz=640,
            device=0,
            verbose=False
        )[0]

        vector = embedding.cpu().numpy().reshape(-1)
        features.append(vector)

    X = np.array(features, dtype=np.float32)

    # Normaliser les vecteurs
    X = X / np.maximum(
        np.linalg.norm(X, axis=1, keepdims=True),
        1e-8
    )

    # 2. Former 16 groupes
    kmeans = KMeans(
        n_clusters=B,
        random_state=SEED,
        n_init=10
    )
    labels = kmeans.fit_predict(X)

    # 3. Choisir l'image la plus proche
    # du centre de chaque groupe
    selected = []

    for cluster in range(B):
        indices = np.where(labels == cluster)[0]
        center = kmeans.cluster_centers_[cluster]

        distances = np.linalg.norm(
            X[indices] - center,
            axis=1
        )

        best_index = indices[np.argmin(distances)]
        selected.append(images[best_index])

    return selected


def main():
    train_dir = DATASET / "images" / "train"
    pool_dir = DATASET / "images" / "pool"

    if not YAML.is_file():
        raise FileNotFoundError("Créer data_kmeans.yaml")

    if count_images(DATASET / "images" / "val") != 20:
        raise RuntimeError("Validation incorrecte")

    if count_images(DATASET / "images" / "test") != 20:
        raise RuntimeError("Test incorrect")

    for cycle in range(1, 5):
        print(f"\n===== K-MEANS CYCLE {cycle} =====")

        previous = (
            RUNS / "train_initial" / "weights" / "best.pt"
            if cycle == 1
            else RUNS / f"kmeans_cycle{cycle-1}"
            / "weights" / "best.pt"
        )

        output = RUNS / f"kmeans_cycle{cycle}"
        best = output / "weights" / "best.pt"

        before = 20 + (cycle - 1) * B
        after = before + B

        n_train = count_images(train_dir)
        n_pool = count_images(pool_dir)

        print("Train :", n_train, "| Pool :", n_pool)

        if n_train + n_pool != 86:
            raise RuntimeError("Nombre total d'images incorrect")

        if not previous.is_file():
            raise FileNotFoundError(previous)

        if best.is_file():
            if n_train < after:
                raise RuntimeError("Modèle existant, dataset incohérent")
            print("Modèle déjà sauvegardé : entraînement ignoré.")
        else:
            if n_train == before and n_pool == 86 - before:
                model = YOLO(str(previous))

                selected = select_images(model, pool_dir)
                print("Images sélectionnées :", len(selected))

                if len(selected) != B:
                    raise RuntimeError("Sélection incomplète")

                # Vérifier les annotations AVANT les déplacements
                for image in selected:
                    label = DATASET / "oracle_labels" / "pool" / (
                        image.stem + ".txt"
                    )
                    if not label.is_file():
                        raise FileNotFoundError(label)

                output.mkdir(parents=True, exist_ok=True)

                # Garder les noms des images sélectionnées
                with open(
                    output / "selected_images.txt",
                    "w",
                    encoding="utf-8"
                ) as f:
                    for image in selected:
                        f.write(image.name + "\n")

                # Déplacer les images et les labels
                for image in selected:
                    label = DATASET / "oracle_labels" / "pool" / (
                        image.stem + ".txt"
                    )

                    shutil.move(str(image), str(train_dir))
                    shutil.move(
                        str(label),
                        str(DATASET / "labels" / "train")
                    )

            elif n_train == after and n_pool == 86 - after:
                print("Sélection déjà effectuée : entraînement repris.")
            else:
                raise RuntimeError(
                    "État inattendu : vérifier le dataset avant de relancer"
                )

            # 4. Entraîner le modèle
            print("Entraînement sur", count_images(train_dir), "images")

            model = YOLO(str(previous))
            model.train(
                data=str(YAML),
                epochs=15
                imgsz=640,
                batch=4,
                device=0,
                workers=0,
                seed=42,
                project=str(RUNS),
                name=f"kmeans_cycle{cycle}",
                exist_ok=True
            )

        if not best.is_file():
            raise RuntimeError(f"best.pt manquant au cycle {cycle}")

        # 5. Évaluer best.pt
        model = YOLO(str(best))
        metrics = model.val(
            data=str(YAML),
            split="val",
            device=0,
            project=str(RUNS),
            name="kmeans_cycle4_30epochs"

        )

        print(
            f"CYCLE {cycle} | "
            f"mAP50={metrics.box.map50:.4f} | "
            f"mAP50-95={metrics.box.map:.4f}"
        )

        # Enregistrer les résultats dans un CSV
        csv_path = RUNS / "kmeans_metrics.csv"

        rows = []
        if csv_path.is_file():
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))

        rows = [r for r in rows if int(r["cycle"]) != cycle]
        rows.append({
            "cycle": cycle,
            "train_images": after,
            "map50": round(metrics.box.map50, 4),
            "map50_95": round(metrics.box.map, 4)
        })
        rows.sort(key=lambda r: int(r["cycle"]))

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "cycle", "train_images", "map50", "map50_95"
                ]
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\nLes 4 cycles K-Means sont terminés !")


if __name__ == "__main__":
    main()
