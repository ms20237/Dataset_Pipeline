import os
import zipfile
import rarfile
import argparse
from tqdm import tqdm
from pathlib import Path
from rich import print

from shuttrix.tasks.utils import parse_print_args


def init():
    """
        Initializes the argument parser for the dataset extraction task.
        This function sets up an argument parser to handle command-line arguments
        for unzipping or unraring dataset archives. It defines the required
        parameters and their descriptions.
    
    """
    parser = argparse.ArgumentParser(description="Unzip or Unrar a dataset.")
    parser.add_argument('--datasets_dir',
                        type=str,
                        required=True,
                        help="Path to the directory containing the dataset archives.")
    
    
    args = parser.parse_args()
    return args 
    
    
@parse_print_args
def run(datasets_dir):
    """
        Extracts all .zip and .rar files in the given directory into subfolders,
        showing progress bars for file-level and archive-level extraction.

        Args:
            datasets_dir (str): Path to the directory containing archive files.
        
    """
    
    print(
        """
            =======================================================================================================
                
                                                UNZIP OR UNRAR DATASET
                                                
            =======================================================================================================
        """
    )
    
    datasets_dir = Path(datasets_dir)
    if not datasets_dir.exists():
        print(f"❌ Directory not found: {datasets_dir}")
        return

    archive_files = list(datasets_dir.glob("*.zip")) + list(datasets_dir.glob("*.rar"))

    if not archive_files:
        print(f"📂 No .zip or .rar files found in '{datasets_dir}'")
        return

    print(f"📦 Found {len(archive_files)} archive(s)\n")

    for archive_path in tqdm(archive_files, desc="Processing archives", unit="archive"):
        out_dir = datasets_dir / archive_path.stem
        os.makedirs(out_dir, exist_ok=True)

        try:
            # for zip files
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    members = zf.namelist()
                    for member in tqdm(members, desc=f"🔓 Extracting {archive_path.name}", unit="file", leave=False):
                        zf.extract(member, out_dir)

            # for rar files
            elif archive_path.suffix == ".rar":
                with rarfile.RarFile(archive_path, 'r') as rf:
                    members = rf.namelist()
                    for member in tqdm(members, desc=f"🔓 Extracting {archive_path.name}", unit="file", leave=False):
                        rf.extract(member, out_dir)

            else:
                print(f"⚠️ Unsupported format: {archive_path.name}")

        except Exception as e:
            print(f"❌ Failed to extract {archive_path.name}: {e}")
        else:
            print(f"✅ Extracted: {archive_path.name}\n")
            
    print("✅ All Done!!!!!")        

    
if __name__ == "__main__":
    args = init()
    run(args.datasets_dir) 
    
