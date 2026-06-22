from pathlib import Path
import json
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(r"E:/henrique/data")
DATASET_ROOT = ROOT / "MITAboveFiveK"

EXPERT = "c"

N_TRAIN = 30
N_VAL = 5
N_TEST = 3

METADATA_URLS = {
    "train": "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main/training.json",
    "val": "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main/validation.json",
    "test": "https://huggingface.co/datasets/yuukicammy/MIT-Adobe-FiveK/raw/main/testing.json",
}

METADATA_FILES = {
    "train": DATASET_ROOT / "training.json",
    "val": DATASET_ROOT / "validation.json",
    "test": DATASET_ROOT / "testing.json",
}

MANIFEST_PATH = DATASET_ROOT / "small_30_5_3_manifest.json"


def download_file(url: str, out_path: Path, timeout: int = 180):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Already exists: {out_path}")
        return

    print(f"Downloading:")
    print(f"  URL : {url}")
    print(f"  SAVE: {out_path}")

    with requests.get(url, stream=True, verify=False, timeout=timeout) as r:
        r.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_metadata(split: str):
    metadata_path = METADATA_FILES[split]

    if metadata_path.exists() and metadata_path.stat().st_size > 0:
        print(f"Metadata already exists: {metadata_path}")
        return

    print(f"Downloading metadata for split={split}")
    download_file(METADATA_URLS[split], metadata_path)


def load_metadata(split: str) -> dict:
    ensure_metadata(split)

    with open(METADATA_FILES[split], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(metadata, dict):
        raise TypeError(
            f"Expected metadata for split={split} to be a dict, "
            f"but got {type(metadata)}"
        )

    return metadata


def camera_folder_from_item(item: dict) -> str:
    camera = item.get("camera", {})

    make = camera.get("make", "Unknown")
    model = camera.get("model", "Unknown")

    return f"{make}_{model}".replace(" ", "_")


def local_dng_path(basename: str, item: dict) -> Path:
    camera_folder = camera_folder_from_item(item)
    return DATASET_ROOT / "raw" / camera_folder / f"{basename}.dng"


def local_tiff_path(basename: str, expert: str) -> Path:
    return DATASET_ROOT / "processed" / f"tiff16_{expert}" / f"{basename}.tif"


def get_dng_url(item: dict) -> str:
    return item["urls"]["dng"]


def get_tiff_url(item: dict, expert: str) -> str:
    return item["urls"]["tiff16"][expert]


def download_split(split: str, n: int) -> list:
    print("\n" + "=" * 80)
    print(f"Preparing split={split}, n={n}")
    print("=" * 80)

    metadata = load_metadata(split)

    # Metadata is a dict:
    # {
    #   "a0001-jmac_DSC1459": {...},
    #   "a0002-...": {...},
    #   ...
    # }
    selected = list(metadata.items())[:n]

    downloaded_items = []

    for i, (basename, item) in enumerate(selected, start=1):
        print(f"\n[{split}] {i}/{n}: {basename}")

        dng_url = get_dng_url(item)
        tiff_url = get_tiff_url(item, EXPERT)

        dng_path = local_dng_path(basename, item)
        tiff_path = local_tiff_path(basename, EXPERT)

        download_file(dng_url, dng_path)
        download_file(tiff_url, tiff_path)

        downloaded_items.append({
            "split": split,
            "basename": basename,
            "dng": str(dng_path),
            "tiff16_c": str(tiff_path),
        })

    return downloaded_items


def main():
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "root": str(DATASET_ROOT),
        "expert": EXPERT,
        "train": download_split("train", N_TRAIN),
        "val": download_split("val", N_VAL),
        "test": download_split("test", N_TEST),
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("Done downloading selected FiveK files.")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print("=" * 80)

    print("\nSummary:")
    print(f"  train: {len(manifest['train'])}")
    print(f"  val  : {len(manifest['val'])}")
    print(f"  test : {len(manifest['test'])}")


if __name__ == "__main__":
    main()