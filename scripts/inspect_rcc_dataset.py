import os
from pathlib import Path
from collections import defaultdict

def inspect_directory(root_dir: Path):
    """
    Recursively inspects the directory structure and prints:
    - folder tree
    - file count per extension
    - subfolders with number of files
    """
    print(f"\n📁 Root directory: {root_dir}\n")

    ext_counter = defaultdict(int)
    folder_summary = {}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_path = Path(dirpath).relative_to(root_dir)
        ext_local = defaultdict(int)
        file_list = []

        for file in filenames:
            ext = Path(file).suffix
            ext_counter[ext] += 1
            ext_local[ext] += 1
            file_list.append(file)

        folder_summary[str(rel_path)] = {
            "n_files": len(filenames),
            "extensions": dict(ext_local),
            "examples": file_list[:500]  # mostra max esempi
        }

    print("📂 Folder overview:\n")
    for folder, summary in folder_summary.items():
        print(f"🗂️  {folder or '.'} — {summary['n_files']} file")
        for ext, count in summary["extensions"].items():
            print(f"   └── {ext or '[no extension]'}: {count} file(s)")
        if summary["examples"]:
            for f in summary["examples"]:
                print(f"      • {f}")
        print()

    print("📊 File count per extension (globale):")
    for ext, count in sorted(ext_counter.items(), key=lambda x: -x[1]):
        print(f"  {ext or '[no extension]'}: {count} file(s)")

    print("\n✅ Done.\n")


if __name__ == "__main__":
    # Modifica questo path con la root reale del tuo dataset
    RCC_DATASET = Path("~/Library/CloudStorage/GoogleDrive-stefano2001roy@gmail.com/Il mio Drive/Colab_Notebooks/RCC_WSIs").expanduser()
    inspect_directory(RCC_DATASET)
