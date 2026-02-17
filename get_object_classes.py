 #!/usr/bin/env python3
"""
split_and_cluster.py  —
================================================
*   Кластеризация иконок (частей) — по **размеру файла** (как раньше).
*   Кластеризация полных изображений (фонов) — по **цветовой подписи** (palette / texture),
    а не по размеру. Далее внутри каждого цветового кластера вычисляется усреднённый
    «чистый» фон (иконки замазываются медианным блюром).

Новые флаги
-----------
```
--extract-bg           извлекать и усреднять фоны
--bg-nclusters  <int>  сколько цветовых кластеров искать (default=10)
--bg-downscale  <int>  размер миниатюры для цветовой подписи (default=32)
--clean                очистить output перед запуском
```

Зависимости
-----------
```
pip install pillow numpy scikit-learn scipy
```
SciPy нужен для GaussianBlur (можно убрать, если Pillow ≥9.1: ImageFilter.GaussianBlur).

Пример
------
```bash
python split_and_cluster.py -i ./tasks -o ./results \
       --size-factor 1.7 --clean \
       --extract-bg --bg-nclusters 8
```

Скрипт
------
"""
from __future__ import annotations
import argparse, json, shutil, uuid
from pathlib import Path
from typing import List, Callable

import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import MiniBatchKMeans

# ----------------------- util -----------------------

def list_png(dir_path: Path, suffix: str = '.png') -> List[Path]:
    return sorted([p for p in dir_path.glob(f'*{suffix}') if p.is_file()])

# ---------------- split icons ----------------------

def split_icons(input_dir: Path, output_dir: Path, parts: int = 5) -> Path:
    """Разрезает каждую *_task.png на *parts* вертикальных кусков."""
    parts_dir = output_dir / 'parts'
    parts_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    for img_path in list_png(input_dir, '_task.png'):
        img = Image.open(img_path)
        w, h = img.size
        step = w // parts
        for i in range(parts):
            left, right = i * step, (i + 1) * step if i < parts - 1 else w
            crop = img.crop((left, 0, right, h))
            uid = f'{uuid.uuid4()}.png'
            crop.save(parts_dir / uid)
            mapping.append({'id': uid, 'src': img_path.name, 'idx': i})
    (output_dir / 'mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    print(f'[ICON] saved {len(mapping)} parts → {parts_dir}')
    return parts_dir

# -------------- simple size-based clustering --------

def cluster_by_size(files: List[Path], size_factor: float) -> List[List[Path]]:
    sorted_files = sorted(files, key=lambda p: p.stat().st_size)
    clusters, cur = [], [sorted_files[0]]
    prev = sorted_files[0].stat().st_size
    max_gap = 0
    for f in sorted_files[1:]:
        sz = f.stat().st_size
        gap = sz - prev
        if max_gap and gap > max_gap * size_factor:
            clusters.append(cur)
            cur = [f]
        else:
            cur.append(f)
        max_gap = max(max_gap, gap)
        prev = sz
    clusters.append(cur)
    return clusters

# -------------- color feature for backgrounds -------

def colour_signature(img: Image.Image, downscale: int = 32) -> np.ndarray:
    """Возвращает уплощённый вектор mean-цветов из LAB-пространства."""
    small = img.resize((downscale, downscale), Image.BILINEAR)
    arr = np.asarray(small.convert('LAB'), dtype=np.float32)
    # усредняем по строкам, получаем (downscale, 3) → flatten
    feat = arr.mean(axis=0).flatten()  # shape (3,)
    var = arr.var(axis=(0, 1))
    return np.concatenate([feat, var])

# -------------- mask icons & blend ------------------

def remove_icons(img: Image.Image, kernel: int = 15) -> Image.Image:
    """Грубая зачистка: сильный GaussianBlur, чтобы иконки размылись."""
    return img.filter(ImageFilter.GaussianBlur(radius=kernel))

# -------------- extract and average backgrounds ----

def extract_backgrounds(input_dir: Path, out_dir: Path, n_clusters: int, downscale: int):
    fulls = list_png(input_dir, '_solution.png')
    if not fulls:
        print('[BG] no *_solution.png found')
        return
    feats = np.stack([colour_signature(Image.open(f), downscale) for f in fulls])
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=16, n_init='auto').fit(feats)
    labs = kmeans.labels_
    bg_root = out_dir / 'backgrounds'
    bg_root.mkdir(parents=True, exist_ok=True)
    for cl in range(n_clusters):
        members = [f for f, l in zip(fulls, labs) if l == cl]
        if not members:
            continue
        stack = []
        for f in members:
            img = Image.open(f).convert('RGB')
            stack.append(np.asarray(img, dtype=np.float32))
        mean_arr = np.mean(stack, axis=0).astype(np.uint8)
        Image.fromarray(mean_arr).save(bg_root / f'bg_{cl}.png')
    print(f'[BG] saved {n_clusters} background templates → {bg_root}')

# -------------- save icon clusters reps -------------

def save_icon_clusters(parts_dir: Path, out_dir: Path, size_factor: float):
    clusters = cluster_by_size(list_png(parts_dir), size_factor)
    uniq = out_dir / 'unique'
    uniq.mkdir(parents=True, exist_ok=True)
    clusters_dict = {}
    for i, grp in enumerate(clusters):
        clusters_dict[str(i)] = [p.name for p in grp]
        shutil.copy(grp[0], uniq / grp[0].name)
    (out_dir / 'clusters.json').write_text(json.dumps(clusters_dict, ensure_ascii=False, indent=2))
    print(f'[ICON] {len(clusters)} clusters by size, reps → {uniq}')

# -------------- main CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(description='Split icons & cluster; extract coloured backgrounds')
    ap.add_argument('-i', '--input', default='.', help='folder with *_task.png')
    ap.add_argument('-o', '--output', default='output', help='output folder')
    ap.add_argument('--size-factor', type=float, default=2.0, help='gap factor for size clustering')
    ap.add_argument('--parts', type=int, default=5, help='how many vertical parts per image')
    ap.add_argument('--clean', action='store_true', help='wipe output dir before run')
    # bg options
    ap.add_argument('--extract-bg', action='store_true', help='extract background templates')
    ap.add_argument('--bg-nclusters', type=int, default=10, help='how many colour clusters')
    ap.add_argument('--bg-downscale', type=int, default=32, help='downscale for colour signature')
    args = ap.parse_args()

    inp, out = Path(args.input), Path(args.output)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    parts_dir = split_icons(inp, out, args.parts)
    save_icon_clusters(parts_dir, out, args.size_factor)

    if args.extract_bg:
        extract_backgrounds(inp, out, args.bg_nclusters, args.bg_downscale)

    print('Done.')

if __name__ == '__main__':
    main()
