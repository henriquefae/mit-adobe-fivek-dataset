from torch.utils.data import DataLoader
from dataset.fivek import MITAboveFiveK


ROOT = r"E:/henrique/data"


def main():
    dataset = MITAboveFiveK(
        root=ROOT,
        split="debug",
        download=True,
        experts=["c"],
    )

    loader = DataLoader(
        dataset,
        batch_size=None,   # Important for this repo
        num_workers=0,
    )

    for i, item in enumerate(loader):
        print("=" * 80)
        print(f"Item {i}")
        print("Keys:", item.keys())

        if "files" in item:
            print("Files:")
            for k, v in item["files"].items():
                print(f"  {k}: {v}")

    print("Done.")


if __name__ == "__main__":
    main()