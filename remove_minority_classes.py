import shutil
from pathlib import Path

def main():
    # Define paths
    dataset_dir = Path("./Dataset/plantwild")
    txt_file_path = Path("./classes_to_remove.txt") # Assuming it's in your root folder
    info_dir = dataset_dir / "info"

    # 1. Parse the text file for class names (ignoring comments and empty lines)
    classes_to_remove = []
    with open(txt_file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                classes_to_remove.append(line)
                
    print(f"Found {len(classes_to_remove)} classes to remove.")

    # 2. Iterate through splits and safely remove ONLY these class folders
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        for cls_name in classes_to_remove:
            target_dir = dataset_dir / split / cls_name
            
            if target_dir.exists() and target_dir.is_dir():
                shutil.rmtree(target_dir)
                print(f"  [-] Removed: {target_dir}")
            else:
                print(f"  [!] Skipped (Not found): {target_dir}")

    # 3. Create info folder and save the artifact
    info_dir.mkdir(parents=True, exist_ok=True)
    destination_txt = info_dir / "classes_to_remove.txt"
    shutil.copy(txt_file_path, destination_txt)
    
    print("\n" + "="*50)
    print(f"Cleanup complete! List saved to: {destination_txt}")
    print("="*50)

if __name__ == "__main__":
    main()