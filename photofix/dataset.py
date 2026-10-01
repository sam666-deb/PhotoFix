"""Training data for the defect analyzer: clean photos + on-the-fly synthetic defects."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from photofix.analyzer import DEFECTS
from photofix.degradations import random_degrade
from photofix.net import make_views, to_tensor

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class DegradedPhotos(Dataset):
    """Each item: (global_view, local_view, presence[7], severity[7]).

    train=True: random composition crop, flip, and resolution, with a fresh random defect mix every epoch.
    train=False: fixed size, and each index is seeded, so the validation set is identical on every run.
    """

    def __init__(self, folder, train: bool, repeat: int = 1, p_clean: float = 0.25, seed: int = 0):
        self.paths = list_images(folder)
        if not self.paths:
            raise FileNotFoundError(f"No images in {folder}. Run scripts/prepare_data.py first.")
        self.train, self.repeat, self.p_clean, self.seed = train, repeat, p_clean, seed

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, idx):
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed * 1_000_003 + idx)
        img = read_rgb(self.paths[idx % len(self.paths)])

        if self.train:
            h, w = img.shape[:2]
            frac = np.sqrt(rng.uniform(0.6, 1.0))
            ch, cw = int(h * frac), int(w * frac)
            top, left = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
            img = img[top : top + ch, left : left + cw]
            if rng.random() < 0.5:
                img = img[:, ::-1]
            side = int(rng.integers(1024, 2049))
        else:
            side = 1536
        scale = side / max(img.shape[:2])
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img)

        degraded, labels = random_degrade(img, rng, p_clean=self.p_clean)
        g, l = make_views(degraded, rng if self.train else None)
        severity = torch.tensor([labels[d] for d in DEFECTS], dtype=torch.float32)
        return to_tensor(g), to_tensor(l), (severity > 0).float(), severity


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class FiveKPairs(Dataset):
    """MIT-Adobe FiveK (input -> Expert C) pairs at 480p, for the learned LUT.

    Each item: (thumb, input, target). The thumb (THUMB², whole frame) is what the LUT predictor sees;
    input/target are the pixels the predicted LUT is scored on. Training uses random crops + flips; since
    a LUT is applied per pixel, a crop changes the content seen, not the color mapping being learned.

    p_finished: fraction of training samples where the input is the expert's *finished* photo and the
    correct output is that same photo. FiveK inputs are all flat, unprocessed camera renders; without
    these samples the model has never seen a finished photo and "fixes" phone JPEGs that are already
    processed (cooling skin tones, adding contrast twice).
    """

    def __init__(self, folder, train: bool, crop: int = 320, p_finished: float = 0.0):
        from photofix.lut import THUMB

        self.inputs = list_images(Path(folder) / "input")
        if not self.inputs:
            raise FileNotFoundError(f"No images in {folder}/input. Run scripts/prepare_fivek.py first.")
        self.targets = [Path(folder) / "target" / p.name for p in self.inputs]
        self.train, self.crop, self.thumb, self.p_finished = train, crop, THUMB, p_finished

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp, tgt = read_rgb(self.inputs[idx]), read_rgb(self.targets[idx])
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < self.p_finished:
                inp = tgt
            h, w = inp.shape[:2]
            frac = np.sqrt(rng.uniform(0.6, 1.0))
            ch, cw = int(h * frac), int(w * frac)
            top, left = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
            inp, tgt = inp[top : top + ch, left : left + cw], tgt[top : top + ch, left : left + cw]
            if rng.random() < 0.5:
                inp, tgt = inp[:, ::-1], tgt[:, ::-1]
            size = (self.crop, self.crop)
            thumb = cv2.resize(inp, (self.thumb, self.thumb), interpolation=cv2.INTER_AREA)
            inp = cv2.resize(np.ascontiguousarray(inp), size, interpolation=cv2.INTER_AREA)
            tgt = cv2.resize(np.ascontiguousarray(tgt), size, interpolation=cv2.INTER_AREA)
        else:
            thumb = cv2.resize(inp, (self.thumb, self.thumb), interpolation=cv2.INTER_AREA)
        return to_tensor(thumb), to_tensor(np.ascontiguousarray(inp)), to_tensor(np.ascontiguousarray(tgt))


class RestorationPatches(Dataset):
    """Clean/degraded patch pairs for the restorer: (degraded, clean), both (3, crop, crop).

    train=True: a random crop of a random photo with fresh damage each time (`per_image` crops are
    drawn per photo per epoch). train=False: one fixed crop per photo with seeded damage, so the
    validation set is identical across runs.
    """

    MARGIN = 16  # extra context so blur kernels near the patch edge see real pixels

    def __init__(self, folder, train: bool, crop: int = 192, per_image: int = 1, seed: int = 0,
                 with_labels: bool = False):
        self.paths = list_images(folder)
        if not self.paths:
            raise FileNotFoundError(f"No images in {folder}.")
        self.train, self.crop, self.per_image, self.seed = train, crop, per_image, seed
        self.with_labels = with_labels

    def __len__(self):
        return len(self.paths) * self.per_image

    def __getitem__(self, idx):
        from photofix.degradations import restoration_degrade

        rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed * 1_000_003 + idx)
        img = read_rgb(self.paths[idx % len(self.paths)])
        size = self.crop + 2 * self.MARGIN
        h, w = img.shape[:2]
        if self.train:
            if rng.random() < 0.5:  # also see detail at phone-photo scale, not only DIV2K's crisp 2K
                scale = rng.uniform(0.5, 1.0)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]
            top, left = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
        else:
            top, left = (h - size) // 2, (w - size) // 2
        patch = np.ascontiguousarray(img[top : top + size, left : left + size])
        if self.train and rng.random() < 0.5:
            patch = np.ascontiguousarray(patch[:, ::-1])
        clean = patch[self.MARGIN : -self.MARGIN, self.MARGIN : -self.MARGIN]
        degraded, labels = restoration_degrade(patch, rng, margin=self.MARGIN)
        if self.with_labels:
            return to_tensor(degraded), to_tensor(np.ascontiguousarray(clean)), labels
        return to_tensor(degraded), to_tensor(np.ascontiguousarray(clean))
