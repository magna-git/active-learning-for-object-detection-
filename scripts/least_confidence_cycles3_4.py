from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path.cwd()
DATASET = ROOT / "dataset_experiences" / "least_confidence"
POOL = DATASET / "images" / "pool"
TRAIN = DATASET / "images" / "train"
LABELS = DATASET / "labels" / "train"
ORACLE = DATASET / "oracle_labels" / "pool"
YAML = ROOT / "data_least_confidence.yaml"
RUNS = ROOT / "runs"
B = 16

# Le cycle 2 vient d'etre termine : c'est notre point de depart.
previous_best = RUNS / "least_confidence_cycle2" / "weights" / "best.pt"


def count_jpg(folder):
    return len(list(folder.glob("*.jpg")))


def main():
    global previous_best

    for cycle in (3, 4):
        expected_train = 52 + (cycle - 3) * B
        expected_pool = 34 - (cycle - 3) * B
        print(f"\n=== LEAST CONFIDENCE : CYCLE {cycle} ===")
        print("Train :", count_jpg(TRAIN), "| Pool :", count_jpg(POOL))

        # Securite : evite d'ajouter 16 images deux fois si on relance le fichier.
        if count_jpg(TRAIN) != expected_train or count_jpg(POOL) != expected_pool:
            raise RuntimeError("Etat inattendu du dataset : ne pas relancer automatiquement la selection.")
        if not previous_best.is_file():
            raise FileNotFoundError(f"Modele introuvable : {previous_best}")
        if not YAML.is_file():
            raise FileNotFoundError(f"YAML introuvable : {YAML}")

        model = YOLO(str(previous_best))
        scored = []
        for image in sorted(POOL.glob("*.jpg")):
            result = model.predict(source=str(image), conf=0.05,
                                   device=0, verbose=False)[0]
            confidences = result.boxes.conf.cpu().numpy()
            score = float(confidences.min()) if len(confidences) else 1.0
            scored.append((image, score))

        # Score faible = une detection peu confiante dans l'image.
        scored.sort(key=lambda item: (item[1], item[0].name))
        selected = scored[:B]
        if len(selected) != B:
            raise RuntimeError("Pas assez d'images dans le Pool.")

        # Verifier tous les labels avant de deplacer le moindre fichier.
        for image, _ in selected:
            label = ORACLE / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Annotation oracle manquante : {label}")

        # Garder une trace des images choisies pour le compte rendu.
        with (RUNS / f"least_confidence_selection_cycle{cycle}.csv").open(
            "w", encoding="utf-8"
        ) as f:
            f.write("image,minimum_confidence\n")
            for image, score in selected:
                f.write(f"{image.name},{score:.6f}\n")

        for image, _ in selected:
            shutil.move(str(image), str(TRAIN / image.name))
            label = ORACLE / f"{image.stem}.txt"
            shutil.move(str(label), str(LABELS / label.name))

        print("Apres selection -> Train :", count_jpg(TRAIN),
              "| Pool :", count_jpg(POOL))
        print("Entrainement sur", count_jpg(TRAIN), "images...")

        # Reprendre le meilleur modele du cycle precedent.
        model.train(data=str(YAML), epochs=15, imgsz=640, batch=4,
                    device=0, workers=0, project=str(RUNS),
                    name=f"least_confidence_cycle{cycle}", seed=42)

        # Le vrai dossier est fourni par Ultralytics (utile s'il ajoute -2).
        saved_dir = Path(model.trainer.save_dir)
        previous_best = saved_dir / "weights" / "best.pt"
        print(f"Cycle {cycle} termine. Meilleur modele : {previous_best}")

        # Evaluer sur les 20 memes images de validation.
        metrics = YOLO(str(previous_best)).val(
            data=str(YAML), split="val", device=0, workers=0,
            project=str(RUNS), name=f"least_confidence_cycle{cycle}_val"
        )
        print(f"Cycle {cycle} | P={metrics.box.mp:.3f} "
              f"R={metrics.box.mr:.3f} "
              f"mAP50={metrics.box.map50:.3f} "
              f"mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
