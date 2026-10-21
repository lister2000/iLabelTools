# train.py
import os
import yaml
import random
import shutil
from ultralytics import YOLO


def build_yolo_dataset(image_dir, label_dir, class_names,
                       out_root="dataset", val_ratio=0.2, seed=42):
    """组织 dataset/ 结构，返回 data.yaml。检测与分割通用。"""
    random.seed(seed)
    img_dir_out = os.path.join(out_root, "images")
    lbl_dir_out = os.path.join(out_root, "labels")
    for split in ("train", "val"):
        os.makedirs(os.path.join(img_dir_out, split), exist_ok=True)
        os.makedirs(os.path.join(lbl_dir_out, split), exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    pairs = []
    for fn in os.listdir(image_dir):
        if not fn.lower().endswith(exts):
            continue
        stem, _ = os.path.splitext(fn)
        lbl_path = os.path.join(label_dir, stem + ".txt")
        if os.path.exists(lbl_path):
            pairs.append((os.path.join(image_dir, fn), lbl_path))

    if not pairs:
        raise RuntimeError("没有找到任何已标注的图片。")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_set = set(p[0] for p in pairs[:n_val])

    for img_path, lbl_path in pairs:
        split = "val" if img_path in val_set else "train"
        shutil.copy2(img_path, os.path.join(img_dir_out, split,
                                            os.path.basename(img_path)))
        shutil.copy2(lbl_path, os.path.join(lbl_dir_out, split,
                                            os.path.basename(lbl_path)))

    data = {
        "path": os.path.abspath(out_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": list(class_names),
    }
    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def train_model(data_yaml_path, task="detect", epochs=100, model_size="n",
                imgsz=640, batch=8, project="runs/train", name="my_model",
                device="cpu", callbacks=None):
    """task: detect 用 yolov8X.pt；segment 用 yolov8X-seg.pt"""
    if task == "segment":
        model_name = f"yolov8{model_size}-seg.pt"
    else:
        model_name = f"yolov8{model_size}.pt"

    model = YOLO(model_name)
    if callbacks:
        for k, fn in callbacks.items():
            model.add_callback(k, fn)

    kwargs = dict(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
        device=device,
        verbose=False,
        amp=False,
        plots=False,
    )
    model.train(**kwargs)
    return os.path.abspath(os.path.join(project, name, "weights", "best.pt"))