
from pathlib import Path
from ultralytics import YOLO
import csv

ROOT = Path.cwd()
RUNS = ROOT / "runs"
YAML = ROOT / "data_kmeans.yaml"

# Modifier ici pour choisir les expériences
EPOCHS_TEST = [30,40, 50, 60]

# Même modèle initial pour tous les tests
MODEL_PATH = RUNS / "kmeans_cycle3" / "weights" / "best.pt"

def main():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    results = []

    for epochs in EPOCHS_TEST:

        print(f"\n===== TEST : {epochs} EPOCHS =====")

        name = f"kmeans_cycle4_{epochs}epochs"
        best_path = RUNS / name / "weights" / "best.pt"

        # Ne pas refaire un entraînement déjà enregistré
        if best_path.is_file():
            print("Modèle déjà présent : validation uniquement.")
        else:
            model = YOLO(str(MODEL_PATH))

            model.train(
                data=str(YAML),
                epochs=epochs,
                imgsz=640,
                batch=4,
                device=0,
                workers=0,
                seed=42,
                project=str(RUNS),
                name=name,
                exist_ok=True
            )

        # Évaluer le meilleur modèle
        model = YOLO(str(best_path))

        metrics = model.val(
            data=str(YAML),
            split="val",
            device=0,
            project=str(RUNS),
            name=f"{name}_val",
            exist_ok=True
        )

        map50 = metrics.box.map50
        map5095 = metrics.box.map

        print(f"mAP50 : {map50:.4f}")
        print(f"mAP50-95 : {map5095:.4f}")

        results.append({
            "epochs": epochs,
            "map50": round(map50, 4),
            "map50_95": round(map5095, 4)
        })

    # Enregistrer le tableau comparatif
    csv_path = RUNS / "kmeans_epochs_comparison.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epochs", "map50", "map50_95"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nTous les tests sont terminés !")
    print("Résultats :", csv_path)


if __name__ == "__main__":
    main()
