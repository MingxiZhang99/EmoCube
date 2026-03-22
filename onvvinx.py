import os
import json
import argparse

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as T
import torchvision.models as models

# cd C:\Users\Lenovo\robot\lab1
# python -u onvvinx.py

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_p.append(pred)
        all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_p)


# -------------------------
# Dataset
# -------------------------
class EmotionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tfms):
        self.paths = df["real_path"].tolist()
        self.labels = df["label"].tolist()
        self.tfms = tfms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.tfms(img)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_real_path", type=str, default="DiffusionEmotion_S/cropped")
    parser.add_argument("--csv_path", type=str, default="DiffusionEmotion_S/dataset_sheet.csv")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out_dir", type=str, default="export_out")
    parser.add_argument("--opset", type=int, default=13)

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 4-class mapping: merge fear + disgust -> fear
    emotion_map4 = {
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "fear",
    }
    classes = ["happy", "sad", "angry", "fear"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # -------------------------
    # Load CSV + build paths
    # -------------------------
    df = pd.read_csv(args.csv_path)
    df["raw_emotion"] = df["subDirectory_filePath"].apply(lambda p: p.split("/")[1].lower())
    df["filename"] = df["subDirectory_filePath"].apply(lambda p: p.split("/")[2])
    df["real_path"] = df.apply(
        lambda r: os.path.join(args.base_real_path, r["raw_emotion"], r["filename"]),
        axis=1,
    )
    df = df[df["real_path"].apply(os.path.exists)].reset_index(drop=True)

    df["mapped_emotion"] = df["raw_emotion"].map(emotion_map4)
    df = df[df["mapped_emotion"].notna()].reset_index(drop=True)
    df["label"] = df["mapped_emotion"].map(class_to_idx).astype(int)

    print("Total images after filtering:", len(df))
    print(df["mapped_emotion"].value_counts())

    # stratified split 70/15/15
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=args.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=args.seed
    )

    # filter bad images
    train_df = train_df[train_df["real_path"].apply(is_valid_image)].reset_index(drop=True)
    val_df = val_df[val_df["real_path"].apply(is_valid_image)].reset_index(drop=True)
    test_df = test_df[test_df["real_path"].apply(is_valid_image)].reset_index(drop=True)

    # -------------------------
    # Preprocess (Unity side must match)
    # -------------------------
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    train_tfms = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tfms = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

    train_ds = EmotionDataset(train_df, train_tfms)
    val_ds = EmotionDataset(val_df, eval_tfms)
    test_ds = EmotionDataset(test_df, eval_tfms)

    # class weights + sampler
    train_labels = train_df["label"].to_numpy()
    class_counts = np.bincount(train_labels, minlength=len(classes))
    class_weights = (class_counts.sum() / (len(classes) * np.maximum(class_counts, 1))).astype(np.float32)

    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

    # -------------------------
    # Model: MobileNetV2 (ONNX/Unity friendlier than v3)
    # -------------------------
    # model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    # model = model.to(device)
    # -------------------------
    # Model: MobileNetV3 Small
    # -------------------------
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    # best_path = os.path.join(args.out_dir, "best_emotion_mnv2.pth")
    best_path = os.path.join(args.out_dir, "best_emotion_mnv3.pth")
    # -------------------------
    # Train
    # -------------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        yv, pv = eval_metrics(model, val_loader, device)
        val_acc = float((yv == pv).mean())

        print(f"Epoch {epoch+1:02d} | loss={total_loss/len(train_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "mean": MEAN,
                "std": STD,
                "arch": "mobilenet_v2",
            }, best_path)
            print("Saved best:", best_path)

    print("Best val acc:", best_val_acc)

    # -------------------------
    # Test Report (optional but useful)
    # -------------------------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    yt, pt = eval_metrics(model, test_loader, device)
    print("\n===== Test Report =====")
    print(classification_report(yt, pt, target_names=classes, digits=4))
    print("macro-F1:", f1_score(yt, pt, average="macro"))

    # -------------------------
    # Export ONNX
    # -------------------------
    UNITY_OPSET = 13

    # onnx_path = os.path.join(args.out_dir, "emotion_mnv2_4cls_unity.onnx")
    onnx_path = os.path.join(args.out_dir, "emotion_mnv3_4cls_unity.onnx")
    # ✅ 固定 batch=1（Unity/Barracuda/InferenceEngine 都更稳）
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    model.eval()

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=UNITY_OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        # ✅ 不要 dynamic_axes（Unity 导入器最怕这个）
        dynamo=False,  # ✅ 强制 legacy exporter，避免新 exporter 的复杂转换
    )

    print("Saved Unity-safe ONNX:", onnx_path)

    # 验证 opset
    import onnx
    m = onnx.load(onnx_path)
    print("Exported opset:", m.opset_import[0].version)
    onnx.checker.check_model(m)
    print("ONNX check OK")

    # onnx_path = os.path.join(args.out_dir, "emotion_mnv2_4cls.onnx")
    # dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    # torch.onnx.export(
    #     model.eval(),
    #     dummy,
    #     onnx_path,
    #     export_params=True,
    #     opset_version=args.opset,
    #     do_constant_folding=True,
    #     input_names=["input"],
    #     output_names=["logits"],
    #     dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    # )
    # print("Saved ONNX:", onnx_path)

    # # ONNX check (does not require onnxruntime)
    # try:
    #     import onnx
    #     onnx_model = onnx.load(onnx_path)
    #     onnx.checker.check_model(onnx_model)
    #     print("ONNX check: OK")
    # except Exception as e:
    #     print("ONNX check: FAILED ->", repr(e))

    # -------------------------
    # Save Unity metadata
    # -------------------------
    labels_path = os.path.join(args.out_dir, "labels.json")
    preprocess_path = os.path.join(args.out_dir, "preprocess.json")

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)

    with open(preprocess_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "img_size": args.img_size,
                "mean": MEAN,
                "std": STD,
                "input_layout": "NCHW",
                "input_dtype": "float32",
                "output": "logits",
                "note": "Unity端预处理: RGB->float[0..1], (x-mean)/std, 变为NCHW"
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Saved:", labels_path)
    print("Saved:", preprocess_path)


if __name__ == "__main__":
    main()
