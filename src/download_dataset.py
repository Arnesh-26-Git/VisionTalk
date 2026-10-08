"""
download_dataset.py
--------------------
Fetches and lays out the Flickr8k dataset so the rest of the pipeline finds
it at the paths defined in config.py:

    data/Images/*.jpg
    data/captions.txt   (image_name,caption  -- one caption per line)

Flickr8k is not hosted on a single stable public URL that never changes, so
this script tries a couple of well-known mirrors and falls back to clear
manual instructions if every automated attempt fails. It never crashes the
rest of the pipeline — it prints what to do next.

Usage:
    python src/download_dataset.py
    python src/download_dataset.py --kaggle          # use Kaggle API instead
"""

import argparse
import os
import shutil
import sys
import zipfile

import requests
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, IMAGES_DIR, CAPTIONS_FILE  # noqa: E402

# Known mirrors. These can go stale — that's exactly why we don't hardcode
# a single silent dependency on them and instead give manual instructions.
MIRRORS = [
    "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
    "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
]

MANUAL_INSTRUCTIONS = f"""
Automatic download did not succeed. Please set the dataset up manually —
it only takes a couple of minutes:

  1. Get the data from ONE of these sources:
       a) Kaggle:  https://www.kaggle.com/datasets/adityajn105/flickr8k
          -> kaggle datasets download -d adityajn105/flickr8k
       b) Jason Brownlee's mirror (images + text in two zips):
          https://github.com/jbrownlee/Datasets  (search "Flickr8k")

  2. Unzip so you end up with:
       {IMAGES_DIR}/<lots of .jpg files>
       {CAPTIONS_FILE}   (a CSV with header "image,caption")

     If your captions come as "Flickr8k.token.txt" (tab separated,
     "image#0<TAB>caption") instead of captions.txt, run:
       python src/download_dataset.py --convert-tokens PATH_TO_TOKEN_FILE

  3. Re-run this script to verify: python src/download_dataset.py --verify
"""


def _download_file(url: str, dest: str) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:  # noqa: BLE001 — we deliberately fall back on ANY failure
        print(f"  ! Failed to download {url}: {e}")
        return False


def _unzip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def convert_token_file(token_path: str):
    """Convert Flickr8k.token.txt (tab-separated) into captions.txt (CSV)."""
    rows = [("image", "caption")]
    with open(token_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, caption = line.split("\t")
            img_name = img_id.split("#")[0]
            rows.append((img_name, caption))

    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        for img_name, caption in rows:
            caption = caption.replace(",", " ")  # keep the CSV simple
            f.write(f"{img_name},{caption}\n")
    print(f"Wrote {len(rows) - 1} caption rows to {CAPTIONS_FILE}")


def verify() -> bool:
    ok = True
    n_images = 0
    if os.path.isdir(IMAGES_DIR):
        n_images = len([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if n_images == 0:
        print(f"  ✗ No images found in {IMAGES_DIR}")
        ok = False
    else:
        print(f"  ✓ Found {n_images} images in {IMAGES_DIR}")

    if not os.path.isfile(CAPTIONS_FILE):
        print(f"  ✗ Captions file missing: {CAPTIONS_FILE}")
        ok = False
    else:
        with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        print(f"  ✓ Captions file has {n_lines} lines: {CAPTIONS_FILE}")

    return ok


def try_kaggle_download():
    """Use the Kaggle API if the user has ~/.kaggle/kaggle.json configured."""
    try:
        os.system(
            "kaggle datasets download -d adityajn105/flickr8k -p "
            f"{DATA_DIR} --unzip"
        )
        # This dataset's zip already unpacks to Images/ + captions.txt at top level
        return verify()
    except Exception as e:  # noqa: BLE001
        print(f"Kaggle download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true", help="Use the Kaggle CLI instead of raw mirrors")
    parser.add_argument("--verify", action="store_true", help="Only check whether the dataset is already in place")
    parser.add_argument("--convert-tokens", metavar="TOKEN_FILE", help="Convert a Flickr8k.token.txt file into captions.txt")
    args = parser.parse_args()

    if args.convert_tokens:
        convert_token_file(args.convert_tokens)
        return

    if args.verify:
        sys.exit(0 if verify() else 1)

    os.makedirs(IMAGES_DIR, exist_ok=True)

    if args.kaggle:
        success = try_kaggle_download()
    else:
        print("Attempting automatic download from known mirrors...")
        success = False
        tmp_dir = os.path.join(DATA_DIR, "_tmp_download")
        os.makedirs(tmp_dir, exist_ok=True)
        for url in MIRRORS:
            dest = os.path.join(tmp_dir, os.path.basename(url))
            if _download_file(url, dest):
                try:
                    _unzip(dest, tmp_dir)
                except zipfile.BadZipFile:
                    continue
        # Best-effort: move anything image-like into IMAGES_DIR
        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.move(os.path.join(root, fn), os.path.join(IMAGES_DIR, fn))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        success = verify()

    if not success:
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    print("\nDataset is ready.")


if __name__ == "__main__":
    main()
