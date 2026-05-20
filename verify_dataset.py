from pathlib import Path

def verify_split(split_name):
    split_dir = Path(f"./Dataset/plantwild/{split_name}")
    if not split_dir.exists():
        print(f"[{split_name.upper()}] Folder not found!")
        return

    classes = [d for d in split_dir.iterdir() if d.is_dir()]
    total_images = sum(1 for f in split_dir.rglob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
    aug_images = sum(1 for f in split_dir.rglob("aug_*"))
    
    print(f"[{split_name.upper()}]")
    print(f"  -> Classes remaining: {len(classes)}")
    print(f"  -> Total images:      {total_images}")
    print(f"  -> Augmented images:  {aug_images}")
    print("-" * 40)

print("=" * 40)
print("DATASET INTEGRITY CHECK")
print("=" * 40)
verify_split("train")
verify_split("val")
verify_split("test")