#!/usr/bin/env python3
"""
Quran Batch Download and Marker Tool
======================================
This script automates downloading the 604 pages of the Quran from Kemenag servers,
queries the Quran.com API to auto-detect starting Surah and Ayat for each page,
caches metadata locally for offline execution, runs auto_marker.py on each page,
and saves the output files into structured directories.

Downloads are fully parallelized using multiple threads for maximum speed.

Directory structure:
- <pages_dir>/QK_XXX.webp               (Raw images)
- <pages_dir>/QK_XXX_highlights.json    (Highlight coordinates)
- <pages_dir>/marker/QK_XXX_highlighted.jpg (Annotated debug images)
"""

import os
import sys
import json
import urllib.request
import subprocess
import argparse
import time
import shutil
from pathlib import Path

# ANSI colors for pretty terminal output
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_MAGENTA = "\033[95m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

def log_info(msg):
    print(f"{C_CYAN}[INFO]{C_RESET} {msg}")

def log_success(msg):
    print(f"{C_GREEN}[SUCCESS]{C_RESET} {msg}")

def log_warn(msg):
    print(f"{C_YELLOW}[WARN]{C_RESET} {msg}")

def log_error(msg):
    print(f"{C_RED}[ERROR]{C_RESET} {msg}")

# URL Templates
IMAGE_URL_TEMPLATE = "https://media.qurankemenag.net/khat2/QK_{page:03d}.webp"
API_URL_TEMPLATE = "https://api.quran.com/api/v4/verses/by_page/{page}?fields=verse_key"

def download_image_worker(page, output_path, retries=3):
    """Downloads the WebP image for a page from the remote server (worker thread)."""
    url = IMAGE_URL_TEMPLATE.format(page=page)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(output_path, "wb") as f:
                    f.write(response.read())
            return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0)
    return False

def fetch_page_metadata_worker(page, retries=3):
    """Fetches complete verse metadata for a page using Quran.com API (worker thread)."""
    url = API_URL_TEMPLATE.format(page=page)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                verses = res_data.get("verses", [])
                if not verses:
                    continue
                
                # Extract first verse info
                first_key = verses[0]["verse_key"]
                surah, ayat = map(int, first_key.split(":"))
                
                # Collect all expected verse keys on this page
                expected_verses = [v["verse_key"] for v in verses]
                
                return {
                    "surah": surah,
                    "start_ayat": ayat,
                    "expected_verses": expected_verses
                }
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0)
    return None

def download_and_cache_page(page, pages_dir, needs_meta, force):
    """Helper target function for ThreadPoolExecutor tasks."""
    img_name = f"QK_{page:03d}.webp"
    img_path = pages_dir / img_name
    
    # 1. Download image if missing or forced
    img_success = True
    if not img_path.exists() or force:
        img_success = download_image_worker(page, img_path)
        
    if not img_success:
        return page, False, None
        
    # 2. Get API metadata if missing or forced
    meta = None
    if needs_meta or force:
        meta = fetch_page_metadata_worker(page)
        if not meta:
            return page, False, None
            
    return page, True, meta

def process_page_worker(page, pages_dir, marker_dir, metadata_cache, python_exe, debug):
    """Worker function to process auto-marker and verification for a single page."""
    img_name = f"QK_{page:03d}.webp"
    img_path = pages_dir / img_name
    
    if not img_path.exists():
        return page, "fail", "image file not found", ""
        
    page_str = str(page)
    meta = metadata_cache.get(page_str)
    if not meta:
        return page, "fail", "metadata cache missing", ""
        
    surah = meta["surah"]
    start_ayat = meta["start_ayat"]
    expected_verses = meta.get("expected_verses", [])
    expected_count = len(expected_verses)
    
    # Extract expected surahs from expected_verses
    expected_surahs = sorted(list(set(int(v.split(":")[0]) for v in expected_verses)))
    
    out_json_path = pages_dir / f"QK_{page:03d}_highlights.json"
    
    cmd = [
        python_exe, "auto_marker.py", str(img_path),
        "--surah", str(surah),
        "--start-ayat", str(start_ayat),
        "--expected-count", str(expected_count),
        "--output", str(out_json_path)
    ]
    if expected_surahs:
        cmd.extend(["--expected-surahs", ",".join(map(str, expected_surahs))])
        
    if debug:
        cmd.append("--debug")
        
    try:
        # Run auto_marker.py and capture its outputs
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout
        
        # Move the generated highlighted image to the marker folder
        marker_img_src = pages_dir / f"QK_{page:03d}_highlighted.jpg"
        marker_img_dst = marker_dir / f"QK_{page:03d}_highlighted.jpg"
        
        if marker_img_src.exists():
            shutil.move(str(marker_img_src), str(marker_img_dst))
            
        # Perform Verification Check against cached Quran.com API data
        if out_json_path.exists():
            with open(out_json_path, 'r') as f:
                output_data = json.load(f)
                
            generated_verses = []
            for v in output_data.get("verses", []):
                generated_verses.append(f"{v['surah']}:{v['ayat']}")
                
            if generated_verses == expected_verses:
                return page, "success", f"exactly {len(expected_verses)} expected verses generated.", stdout
            else:
                mismatch_detail = (
                    f"Expected ({len(expected_verses)}): {', '.join(expected_verses)} | "
                    f"Generated ({len(generated_verses)}): {', '.join(generated_verses)}"
                )
                return page, "mismatch", mismatch_detail, stdout
        else:
            return page, "fail", "output JSON highlights file not found after running auto_marker.py", stdout
            
    except subprocess.CalledProcessError as e:
        return page, "fail", f"Error executing auto_marker.py:\n{e.stderr}", ""
    except Exception as e:
        return page, "fail", f"Unexpected error: {e}", ""

def main():
    parser = argparse.ArgumentParser(description="Quran Parallel Batch Download, Auto-Marker, and API Verifier Tool")
    parser.add_argument('--pages-dir', type=str, default='pages',
                        help="Target directory for downloaded pages and output files")
    parser.add_argument('--start-page', type=int, default=1,
                        help="Starting page number (1-604)")
    parser.add_argument('--end-page', type=int, default=604,
                        help="Ending page number (1-604)")
    parser.add_argument('--threads', type=int, default=10,
                        help="Number of concurrent download threads (default: 10)")
    parser.add_argument('--process-threads', type=int, default=4,
                        help="Number of concurrent processing threads for auto-marker (default: 4)")
    parser.add_argument('--download-only', action='store_true',
                        help="Only download raw images and fetch metadata, skip running the auto-marker")
    parser.add_argument('--debug', action='store_true',
                        help="Run auto-marker in debug mode to save coordinate projections")
    parser.add_argument('--force', action='store_true',
                        help="Force re-download and re-marking of existing pages")
    parser.add_argument('--stop-on-fail', action='store_true',
                        help="Stop processing immediately if a page fails verification or execution")
    args = parser.parse_args()

    # Ensure directories exist
    pages_dir = Path(args.pages_dir)
    marker_dir = pages_dir / "marker"
    
    pages_dir.mkdir(parents=True, exist_ok=True)
    marker_dir.mkdir(parents=True, exist_ok=True)

    # Load / Initialize metadata cache
    cache_path = pages_dir / "page_metadata.json"
    metadata_cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                metadata_cache = json.load(f)
            log_info(f"Loaded {len(metadata_cache)} cached page configurations.")
        except Exception as e:
            log_warn(f"Could not load metadata cache file: {e}. Reinitializing.")

    # Load manual page metadata overrides if they exist
    overrides_path = pages_dir / "page_metadata_overrides.json"
    overrides = {}
    if overrides_path.exists():
        try:
            with open(overrides_path, 'r') as f:
                overrides = json.load(f)
            log_info(f"Loaded {len(overrides)} manual page metadata overrides.")
            for p_str, o_meta in overrides.items():
                if p_str in metadata_cache:
                    metadata_cache[p_str] = {**metadata_cache[p_str], **o_meta}
                else:
                    metadata_cache[p_str] = o_meta
        except Exception as e:
            log_warn(f"Could not load metadata overrides file: {e}")

    # Locate the python interpreter inside the current virtual environment if possible
    python_exe = sys.executable
    venv_python = Path(__file__).parent / "venv" / "bin" / "python"
    if venv_python.exists():
        python_exe = str(venv_python)

    # Track processing status
    processed_count = 0
    failed_count = 0
    download_failed_count = 0
    verification_failures = []

    # 1. Check which pages need to be downloaded or updated
    pages_to_download = []
    for page in range(args.start_page, args.end_page + 1):
        img_path = pages_dir / f"QK_{page:03d}.webp"
        page_str = str(page)
        meta = metadata_cache.get(page_str)
        
        needs_img = not img_path.exists() or args.force
        needs_meta = not meta or "expected_verses" not in meta or args.force
        
        if needs_img or needs_meta:
            pages_to_download.append((page, needs_meta))

    # 2. Run Parallel Downloads
    if pages_to_download:
        log_info(f"Starting parallel download/fetch of {len(pages_to_download)} pages using {args.threads} threads...")
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            future_to_page = {
                executor.submit(download_and_cache_page, page, pages_dir, needs_meta, args.force): page
                for page, needs_meta in pages_to_download
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                completed += 1
                percent = (completed / len(pages_to_download)) * 100
                try:
                    page, success, meta = future.result()
                    if success:
                        if meta:
                            metadata_cache[str(page)] = meta
                        log_success(f"[{completed:3d}/{len(pages_to_download):3d}] ({percent:3.0f}%) Page {page:03d} successfully downloaded and cached.")
                    else:
                        log_error(f"[{completed:3d}/{len(pages_to_download):3d}] ({percent:3.0f}%) Page {page:03d} failed download/cache.")
                        download_failed_count += 1
                except Exception as e:
                    log_error(f"[{completed:3d}/{len(pages_to_download):3d}] ({percent:3.0f}%) Page {page:03d} generated exception: {e}")
                    download_failed_count += 1
                    
        # Re-apply overrides before writing to cache file
        for p_str, o_meta in overrides.items():
            if p_str in metadata_cache:
                metadata_cache[p_str] = {**metadata_cache[p_str], **o_meta}
            else:
                metadata_cache[p_str] = o_meta
                
        # Write metadata cache back to disk
        with open(cache_path, 'w') as f:
            json.dump(metadata_cache, f, indent=2)
        log_success("Saved page metadata cache locally.")
    else:
        log_info("All images and metadata are already cached. Skipping download phase.")

    if args.download_only:
        print(f"\n{C_BOLD}{C_GREEN}=================================================={C_RESET}")
        print(f"{C_BOLD}  Download Phase Summary{C_RESET}")
        print(f"{C_BOLD}{C_GREEN}=================================================={C_RESET}")
        print(f"  Total Checked Pages   : {args.end_page - args.start_page + 1}")
        print(f"  Downloaded / Verified : {args.end_page - args.start_page + 1 - download_failed_count}")
        print(f"  Failed Downloads      : {C_RED if download_failed_count > 0 else C_RESET}{download_failed_count}{C_RESET}")
        print(f"{C_BOLD}{C_GREEN}=================================================={C_RESET}")
        return

    # 3. Parallel or Sequential Marking and Verification (fully offline using the local cache)
    pages_to_process = list(range(args.start_page, args.end_page + 1))
    
    if args.process_threads > 1:
        log_info(f"Starting parallel processing of {len(pages_to_process)} pages using {args.process_threads} threads...")
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.process_threads) as executor:
            future_to_page = {
                executor.submit(process_page_worker, page, pages_dir, marker_dir, metadata_cache, python_exe, args.debug): page
                for page in pages_to_process
            }
            
            should_stop = False
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                
                # If a stop was triggered, skip remaining futures
                if should_stop:
                    continue
                    
                print(f"\n{C_BOLD}{C_MAGENTA}--- Processing Page {page:03d} (Parallel) ---{C_RESET}")
                try:
                    page, status, detail, stdout = future.result()
                    if stdout:
                        print(stdout)
                        
                    if status == "success":
                        log_success(f"Verification passed: {detail}")
                        log_success(f"Page {page} processed successfully.")
                        processed_count += 1
                    elif status == "mismatch":
                        log_warn(f"Verification mismatch on Page {page}: {detail}")
                        verification_failures.append({
                            "page": page,
                            "detail": detail
                        })
                        processed_count += 1
                        if args.stop_on_fail:
                            log_error(f"Stopping execution on Page {page} mismatch due to --stop-on-fail.")
                            should_stop = True
                            executor.shutdown(wait=False)
                    else:  # fail
                        log_error(f"Page {page} failed: {detail}")
                        failed_count += 1
                        if args.stop_on_fail:
                            log_error(f"Stopping execution on Page {page} failure due to --stop-on-fail.")
                            should_stop = True
                            executor.shutdown(wait=False)
                            
                except Exception as e:
                    log_error(f"Page {page} raised exception: {e}")
                    failed_count += 1
                    if args.stop_on_fail:
                        log_error(f"Stopping execution on Page {page} exception due to --stop-on-fail.")
                        should_stop = True
                        executor.shutdown(wait=False)
            
            if should_stop:
                log_error("Processing aborted due to failure/mismatch.")
    else:
        # Sequential processing
        for page in pages_to_process:
            print(f"\n{C_BOLD}{C_MAGENTA}--- Processing Page {page:03d} ---{C_RESET}")
            page, status, detail, stdout = process_page_worker(page, pages_dir, marker_dir, metadata_cache, python_exe, args.debug)
            if stdout:
                print(stdout)
                
            if status == "success":
                log_success(f"Verification passed: {detail}")
                log_success(f"Page {page} processed successfully.")
                processed_count += 1
            elif status == "mismatch":
                log_warn(f"Verification mismatch on Page {page}: {detail}")
                verification_failures.append({
                    "page": page,
                    "detail": detail
                })
                processed_count += 1
                if args.stop_on_fail:
                    log_error(f"Stopping execution on Page {page} mismatch due to --stop-on-fail.")
                    break
            else:  # fail
                log_error(f"Page {page} failed: {detail}")
                failed_count += 1
                if args.stop_on_fail:
                    log_error(f"Stopping execution on Page {page} failure due to --stop-on-fail.")
                    break

    # Print summary statistics
    print(f"\n{C_BOLD}{C_GREEN}=================================================={C_RESET}")
    print(f"{C_BOLD}  Batch Processing Summary{C_RESET}")
    print(f"{C_BOLD}{C_GREEN}=================================================={C_RESET}")
    print(f"  Total Requested Pages : {args.end_page - args.start_page + 1}")
    print(f"  Successfully Processed: {C_GREEN}{processed_count}{C_RESET}")
    print(f"  Failed                : {C_RED if failed_count > 0 else C_RESET}{failed_count}{C_RESET}")
    print(f"  Mismatched / Warnings : {C_YELLOW if len(verification_failures) > 0 else C_RESET}{len(verification_failures)}{C_RESET}")
    print(f"{C_BOLD}{C_GREEN}=================================================={C_RESET}")

    if verification_failures:
        print(f"\n{C_BOLD}{C_YELLOW}Verification Mismatches Detail:{C_RESET}")
        for failure in verification_failures:
            print(f"  - {C_BOLD}Page {failure['page']:03d}{C_RESET}: {failure['detail']}")
        print(f"{C_BOLD}{C_YELLOW}=================================================={C_RESET}")

if __name__ == "__main__":
    main()
