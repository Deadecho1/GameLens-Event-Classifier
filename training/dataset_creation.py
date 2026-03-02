import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import albumentations as A

# -----------------------------
# Config
# -----------------------------
@dataclass
class SplitConfig:
    seed: int = 1337
    # default ratios (used for medium/large classes)
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    test_ratio: float = 0.15

    # For tiny classes, force at least these many samples into valid/test (when possible).
    min_valid: int = 1
    min_test: int = 1

    # If a class is very small, don't try to allocate too much to val/test.
    # Example: count=2 -> keep all in train (no val/test)
    min_total_for_splitting: int = 3


@dataclass
class AugmentConfig:
    seed: int = 1337

    # Only augment classes with train_count < target_count
    # You can tune this easily:
    # - If you want to "slightly boost" minorities: set target_count ~ 150-250
    # - If you want to match big class sizes: set higher (but be careful about overfitting)
    target_count: int = 150

    # Safety cap: never generate more than this many augmented images per class
    max_aug_per_class: int = 2000

    # Image extensions to consider
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    # Aug output filename suffix
    aug_prefix: str = "aug"


def build_augmenter() -> A.Compose:
    """
    Augmentations that usually make sense for game screenshots:
    - mild geometric jitter
    - mild brightness/contrast/gamma
    - mild blur/noise/compression
    Keep it moderate to avoid breaking UI text too much.
    """
    return A.Compose(
        [
            A.OneOf(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.03, scale_limit=0.05, rotate_limit=3,
                        border_mode=cv2.BORDER_REFLECT_101, p=1.0
                    ),
                    A.Perspective(scale=(0.02, 0.05), p=1.0),
                ],
                p=0.55,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                    A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0),
                ],
                p=0.55,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
                    A.ImageCompression(quality_lower=60, quality_upper=95, p=1.0),
                ],
                p=0.35,
            ),
        ]
    )


# -----------------------------
# Helpers
# -----------------------------
def list_class_images(class_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_files(files: List[Path], dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for src in files:
        dst = dst_dir / src.name
        # avoid accidental overwrite
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            i = 1
            while True:
                candidate = dst_dir / f"{stem}__{i}{suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                i += 1
        shutil.copy2(src, dst)


def per_class_split(
    files: List[Path],
    cfg: SplitConfig,
    rng: random.Random,
) -> Tuple[List[Path], List[Path], List[Path]]:
    n = len(files)
    if n < cfg.min_total_for_splitting:
        # Too small to split reliably -> keep all in train
        return files, [], []

    shuffled = files[:]
    rng.shuffle(shuffled)

    # initial allocation by ratios
    n_valid = int(round(n * cfg.valid_ratio))
    n_test = int(round(n * cfg.test_ratio))

    # enforce minimums (when possible)
    # leave at least 1 for train
    if n >= (1 + cfg.min_valid + cfg.min_test):
        n_valid = max(n_valid, cfg.min_valid)
        n_test = max(n_test, cfg.min_test)

    # prevent taking too many from train
    if n_valid + n_test >= n:
        # shrink valid/test to fit
        overflow = (n_valid + n_test) - (n - 1)
        # reduce from the larger of valid/test
        while overflow > 0 and (n_valid > 0 or n_test > 0):
            if n_valid >= n_test and n_valid > 0:
                n_valid -= 1
            elif n_test > 0:
                n_test -= 1
            overflow -= 1

    n_train = n - n_valid - n_test
    train_files = shuffled[:n_train]
    valid_files = shuffled[n_train:n_train + n_valid]
    test_files = shuffled[n_train + n_valid:]
    return train_files, valid_files, test_files


def augment_class_to_target(
    train_class_dir: Path,
    augmenter: A.Compose,
    cfg: AugmentConfig,
) -> int:
    """
    Creates augmented images in train_class_dir until target_count is reached.
    Returns how many augmented images were created.
    """
    rng = random.Random(cfg.seed + hash(train_class_dir.name) % 10_000)
    src_files = list_class_images(train_class_dir, cfg.exts)
    if not src_files:
        return 0

    current = len(src_files)
    if current >= cfg.target_count:
        return 0

    needed = min(cfg.target_count - current, cfg.max_aug_per_class)
    created = 0

    for i in range(needed):
        src = rng.choice(src_files)
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue

        aug = augmenter(image=img)["image"]

        # save as PNG to avoid JPEG artifacts compounding
        out_name = f"{src.stem}__{cfg.aug_prefix}{i:05d}.png"
        out_path = train_class_dir / out_name

        # avoid collision
        if out_path.exists():
            j = 1
            while True:
                alt = train_class_dir / f"{src.stem}__{cfg.aug_prefix}{i:05d}__{j}.png"
                if not alt.exists():
                    out_path = alt
                    break
                j += 1

        ok = cv2.imwrite(str(out_path), aug)
        if ok:
            created += 1

    return created


# -----------------------------
# Main pipeline
# -----------------------------
def split_and_augment(
    dataset_root: Path,
    out_root: Path,
    split_cfg: SplitConfig,
    aug_cfg: AugmentConfig,
    per_class_aug_targets: Dict[str, int] | None = None,
) -> None:
    """
    dataset_root should contain: train/<class_name>/...
    out_root will be created with: train/, valid/, test/ each with class subfolders.
    """
    in_train = dataset_root / "train"
    if not in_train.exists():
        raise FileNotFoundError(f"Expected {in_train} to exist.")

    class_dirs = [d for d in in_train.iterdir() if d.is_dir()]
    class_dirs.sort(key=lambda p: p.name)

    rng = random.Random(split_cfg.seed)

    # 1) Split and copy
    print("=== Splitting into train/valid/test ===")
    split_stats = {}

    for cdir in class_dirs:
        cname = cdir.name
        files = list_class_images(cdir, aug_cfg.exts)
        train_files, valid_files, test_files = per_class_split(files, split_cfg, rng)

        dst_train = out_root / "train" / cname
        dst_valid = out_root / "valid" / cname
        dst_test = out_root / "test" / cname

        copy_files(train_files, dst_train)
        copy_files(valid_files, dst_valid)
        copy_files(test_files, dst_test)

        split_stats[cname] = (len(train_files), len(valid_files), len(test_files))
        print(f"{cname:15s}  train={len(train_files):4d}  valid={len(valid_files):4d}  test={len(test_files):4d}")

    # 2) Augment minority classes in TRAIN only
    print("\n=== Augmenting minority classes (train only) ===")
    augmenter = build_augmenter()

    for cname, (n_train, _, _) in split_stats.items():
        # target count can be per-class override or global
        target = per_class_aug_targets.get(cname, aug_cfg.target_count) if per_class_aug_targets else aug_cfg.target_count

        train_class_dir = out_root / "train" / cname
        # refresh count (because copy stage may have adjusted names)
        current_train = len(list_class_images(train_class_dir, aug_cfg.exts))

        if current_train < target:
            # apply target per class
            local_aug_cfg = AugmentConfig(**{**aug_cfg.__dict__, "target_count": target})
            created = augment_class_to_target(train_class_dir, augmenter, local_aug_cfg)
            new_count = len(list_class_images(train_class_dir, aug_cfg.exts))
            print(f"{cname:15s}  {current_train:4d} -> {new_count:4d}  (+{created})")
        else:
            print(f"{cname:15s}  {current_train:4d} (no aug)")

    print("\nDone.")
    print(f"Output dataset written to: {out_root}")


if __name__ == "__main__":
    # -----------------------------
    # EDIT THESE PATHS
    # -----------------------------
    DATASET_ROOT = Path(r"path/to/clean/dataset")   # contains train/<class>/*
    OUT_ROOT = Path(r"new/dataset/path") # will be created/filled

    # -----------------------------
    # Split config
    # -----------------------------
    split_cfg = SplitConfig(
        seed=1337,
        train_ratio=0.70,
        valid_ratio=0.15,
        test_ratio=0.15,
        min_valid=1,
        min_test=1,
        min_total_for_splitting=3,
    )

    # -----------------------------
    # Augment config
    # -----------------------------
    aug_cfg = AugmentConfig(
        seed=1337,
        target_count=150,         
        max_aug_per_class=2000,
        exts=(".jpg", ".jpeg", ".png", ".webp"),
        aug_prefix="aug",
    )

    # OPTIONAL: per-class targets (recommended for your specific counts)
    # - keep big classes as-is
    # - boost minorities to something reasonable but not insane
    per_class_targets = {
        "none": 0,          # 0 means "use global", but we’ll just set it high and it won’t augment anyway
        "boss-fight": 0,
        "choice": 0,
        "dialog": 0,
        "notification": 0,
        "boss-kill": 100,
        "victory": 50,
        "enter-level": 50,
    }

    # If you put 0, treat as "global"
    per_class_targets = {k: (aug_cfg.target_count if v == 0 else v) for k, v in per_class_targets.items()}

    split_and_augment(
        dataset_root=DATASET_ROOT,
        out_root=OUT_ROOT,
        split_cfg=split_cfg,
        aug_cfg=aug_cfg,
        per_class_aug_targets=per_class_targets,
    )