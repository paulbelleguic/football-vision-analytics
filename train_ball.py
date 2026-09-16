from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    project_dir = Path(__file__).resolve().parent
    dataset_dir = project_dir / "datasets" / "football_ball"

    with (dataset_dir / "data.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["path"] = dataset_dir.as_posix()
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    local_config = dataset_dir / "data_local.yaml"
    with local_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True)

    model = YOLO("yolo26n.pt")

    model.train(
        data=str(local_config),
        epochs=3,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=str(project_dir / "runs" / "train"),
        name="ball_smoke_test",
    )


if __name__ == "__main__":
    main()