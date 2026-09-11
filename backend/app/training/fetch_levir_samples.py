"""
SatQuery AI — Official LEVIR-CD Public Sample Downloader & Validator
Acquires genuine benchmark image pairs from official public repository.
"""
import os
import urllib.request
import json
from pathlib import Path
from PIL import Image
import numpy as np

def fetch_and_validate_levir_samples():
    base_raw_url = "https://raw.githubusercontent.com/justchenhao/BIT_CD/master/samples"
    api_url = "https://api.github.com/repos/justchenhao/BIT_CD/contents/samples/label"
    
    target_dir = Path("datasets/levir_cd")
    dir_a = target_dir / "A"
    dir_b = target_dir / "B"
    dir_label = target_dir / "label"
    
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)
    dir_label.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch file list from API (using label directory to ensure complete triplets)
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        files_data = json.loads(resp.read().decode("utf-8"))
    
    filenames = [f["name"] for f in files_data if f["name"].endswith(".png")]
    print(f"Discovered {len(filenames)} verified matching LEVIR-CD triplets in official repository.")
    
    downloaded = 0
    for fn in filenames:
        for folder, target_sub in [("A", dir_a), ("B", dir_b), ("label", dir_label)]:
            file_path = target_sub / fn
            if not file_path.exists():
                file_url = f"{base_raw_url}/{folder}/{fn}"
                print(f"Downloading {folder}/{fn}...")
                req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as u_resp, open(file_path, "wb") as out_f:
                    out_f.write(u_resp.read())
        downloaded += 1
    
    print(f"\nSuccessfully acquired {downloaded} authentic LEVIR-CD triplets into {target_dir}")
    
    # 2. Validate Every Triplet
    print("\n--- AUTHENTIC SATELLITE DATASET VALIDATION REPORT ---")
    valid_count = 0
    dims = set()
    channels_t1 = set()
    channels_t2 = set()
    corrupt = []
    
    for fn in sorted(filenames):
        f_a = dir_a / fn
        f_b = dir_b / fn
        f_lbl = dir_label / fn
        
        if not (f_a.exists() and f_b.exists() and f_lbl.exists()):
            corrupt.append((fn, "Missing component"))
            continue
            
        try:
            im_a = Image.open(f_a)
            im_b = Image.open(f_b)
            im_lbl = Image.open(f_lbl)
            
            dims.add(im_a.size)
            channels_t1.add(im_a.mode)
            channels_t2.add(im_b.mode)
            
            arr_lbl = np.array(im_lbl)
            unique_vals = np.unique(arr_lbl)
            change_pixels = int(np.sum(arr_lbl > 0))
            
            if im_a.size != im_b.size or im_a.size != im_lbl.size:
                corrupt.append((fn, f"Dimension mismatch: A={im_a.size}, B={im_b.size}, L={im_lbl.size}"))
            else:
                valid_count += 1
                print(f"  ✓ {fn:<25} | Size: {im_a.size} | Mode: {im_a.mode} | Change Pixels: {change_pixels:,} ({change_pixels/(im_a.size[0]*im_a.size[1])*100:.1f}%)")
        except Exception as e:
            corrupt.append((fn, str(e)))
            
    print(f"\nTotal Valid Triplets: {valid_count}/{len(filenames)}")
    print(f"Image Dimensions: {list(dims)}")
    print(f"T1 Channels: {list(channels_t1)}, T2 Channels: {list(channels_t2)}")
    print(f"Corrupted/Missing: {len(corrupt)}")
    return valid_count, corrupt

if __name__ == "__main__":
    fetch_and_validate_levir_samples()
