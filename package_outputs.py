import os
import json
import zipfile
import shutil

def main():
    print("Packaging outputs into quran_markers_data.zip...")
    staging = "quran_data_staging"
    os.makedirs(staging, exist_ok=True)
    
    # Copy metadata files
    for meta_file in ["pages/page_metadata.json", "pages/page_metadata_overrides.json"]:
        if os.path.exists(meta_file):
            shutil.copy2(meta_file, os.path.join(staging, os.path.basename(meta_file)))
        else:
            # Fallback to root directory if not inside pages/
            root_meta = os.path.basename(meta_file)
            if os.path.exists(root_meta):
                shutil.copy2(root_meta, os.path.join(staging, root_meta))
            
    # Process pages
    for i in range(1, 605):
        padded = f"{i:03d}"
        src_webp = f"pages/QK_{padded}.webp"
        src_json = f"pages/QK_{padded}_highlights.json"
        
        dst_webp = os.path.join(staging, f"{padded}.webp")
        dst_json = os.path.join(staging, f"{padded}.json")
        
        if os.path.exists(src_webp):
            shutil.copy2(src_webp, dst_webp)
        if os.path.exists(src_json):
            with open(src_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["image"] = f"{padded}.webp"
            with open(dst_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
    # Create zip file
    zip_filename = "quran_markers_data.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(staging):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, staging)
                zipf.write(filepath, arcname)
                
    # Clean up staging directory
    shutil.rmtree(staging)
    print(f"Successfully packaged {zip_filename}!")

if __name__ == "__main__":
    main()
