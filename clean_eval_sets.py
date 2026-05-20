from pathlib import Path

def clean_split(split_name):
    split_dir = Path(f"./Dataset/plantwild/{split_name}")
    deleted = 0
    # Hunts down and deletes only files with the "aug_" prefix
    for aug_file in split_dir.rglob("aug_*"):
        aug_file.unlink()
        deleted += 1
    print(f"Deleted {deleted} augmented files from {split_name}/. Real images preserved.")

clean_split("test")
clean_split("val")