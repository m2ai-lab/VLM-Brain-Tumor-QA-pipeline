import os
import requests
import json
import sys

def download_ucsf_pdgm(target_dir):
    """
    Downloads UCSF-PDGM NIfTI images from the data-nih/tcia GitHub mirror.
    This is an automated alternative to the 142GB manual TCIA download.
    """
    print(f"--- Automated UCSF-PDGM Image Download to {target_dir} ---")
    
    # Ensure directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # GitHub API URL for the ucsf-pdgm release
    api_url = "https://api.github.com/repos/data-nih/tcia/releases/tags/ucsf-pdgm"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching release info: {e}")
        return

    assets = data.get("assets", [])
    if not assets:
        print("No assets found in the release.")
        return

    print(f"Found {len(assets)} image assets. Starting download...")
    
    for i, asset in enumerate(assets):
        filename = asset["name"]
        download_url = asset["browser_download_url"]
        dest_path = os.path.join(target_dir, filename)
        
        if os.path.exists(dest_path):
            # Skipping existing files to allow resuming
            continue
            
        print(f"[{i+1}/{len(assets)}] Downloading {filename}...")
        try:
            asset_resp = requests.get(download_url, stream=True)
            asset_resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in asset_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

    print("Download complete.")

if __name__ == "__main__":
    # Default to datasets/ucsf-pdgm-images relative to project root
    # We expect this to be called from the project root
    target = os.path.join("datasets", "ucsf-pdgm-images")
    download_ucsf_pdgm(target)
