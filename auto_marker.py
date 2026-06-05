#!/usr/bin/env python3
"""
Quran Verse Auto-Marker Tool
==============================
Automatically detects verse markers (circled numbers) in Quran page images,
determines text line positions, and generates highlight regions per verse.

Uses HoughCircles for marker detection with text-line proximity filtering.

Usage:
    python auto_marker.py QK_004.jpg --surah 2 --start-ayat 17
    python auto_marker.py QK_004.jpg --surah 2 --start-ayat 17 --end-ayat 24
"""

import cv2
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path

# ============================================================
# CONSTANTS
# ============================================================

BLOCK_HEIGHT = 68           # Fixed height for each highlight block (px)
CONTENT_LEFT = 58           # Left margin of text content area (px)
CONTENT_RIGHT = 772         # Right margin of text content area (px)
CONTENT_TOP = 65            # Top margin (below header bar, excludes ornaments)
CONTENT_BOTTOM = 1220       # Bottom margin
MARKER_RADIUS = 22          # Approximate radius of verse markers (px)

VERSE_COLORS = [
    (255, 180, 50), (50, 220, 130), (80, 130, 255), (220, 100, 255),
    (50, 255, 255), (255, 100, 100), (100, 255, 200), (200, 50, 255),
    (255, 220, 100), (100, 180, 255), (180, 255, 100), (255, 100, 200),
]


# ============================================================
# TEXT LINE DETECTION
# ============================================================

def detect_text_lines(img, debug=False, page=0):
    """
    Detect text line center positions using horizontal projection profile.
    Returns list of y_center values.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    content = gray[CONTENT_TOP:CONTENT_BOTTOM, CONTENT_LEFT:CONTENT_RIGHT]

    _, binary = cv2.threshold(content, 180, 255, cv2.THRESH_BINARY_INV)
    projection = np.sum(binary, axis=1).astype(float)

    kernel = np.ones(7) / 7
    projection = np.convolve(projection, kernel, mode='same')

    pmax = np.max(projection)
    if pmax == 0:
        return []
    projection /= pmax

    # Peak detection
    min_height = 0.30
    min_distance = 60 if page == 568 else 70

    peaks = []
    for i in range(1, len(projection) - 1):
        if (projection[i] > min_height and
            projection[i] >= projection[i - 1] and
            projection[i] >= projection[i + 1]):
            peaks.append(i)

    filtered = []
    for p in peaks:
        if filtered:
            prev = filtered[-1]
            if p - prev < min_distance:
                # Check for a deep valley between prev and p
                valley_val = np.min(projection[prev:p+1])
                if valley_val < 0.10:
                    # Deep valley indicates separate lines (e.g. text adjacent to header border)
                    filtered.append(p)
                else:
                    # No deep valley: merge peaks, keeping the higher one
                    if projection[p] > projection[prev]:
                        filtered[-1] = p
            else:
                filtered.append(p)
        else:
            filtered.append(p)

    line_centers = [p + CONTENT_TOP for p in filtered]

    if debug:
        debug_img = img.copy()
        for i, yc in enumerate(line_centers):
            yt = yc - BLOCK_HEIGHT // 2
            yb = yc + BLOCK_HEIGHT // 2
            cv2.rectangle(debug_img, (CONTENT_LEFT, yt), (CONTENT_RIGHT, yb), (0, 255, 0), 1)
            cv2.putText(debug_img, f"L{i}", (CONTENT_LEFT - 25, yc + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.imwrite("debug_lines.jpg", debug_img)

    return line_centers


def validate_circle(gray_img, cx, cy, r, thresh_circ=0.5):
    """
    Validate if a candidate circle is a real verse marker by checking
    if it contains a centered contour with high circularity.
    """
    pad = 5
    y1 = max(0, cy - r - pad)
    y2 = min(gray_img.shape[0], cy + r + pad)
    x1 = max(0, cx - r - pad)
    x2 = min(gray_img.shape[1], cx + r + pad)

    crop = gray_img[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    # Filter out low-contrast background or solid margin ornaments
    if int(np.max(crop)) - int(np.min(crop)) < 100:
        return False

    _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    crop_h, crop_w = crop.shape[:2]
    center_x, center_y = crop_w / 2, crop_h / 2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(cnt)
        cx_cnt = x + w / 2
        cy_cnt = y + h / 2
        dist_from_center = np.sqrt((cx_cnt - center_x)**2 + (cy_cnt - center_y)**2)

        # Real verse markers have size of around 2*r (typically 25 to 50 pixels)
        # and should be centered in the crop box
        if 20 < w < 52 and 20 < h < 52 and dist_from_center < 10:
            if circularity > thresh_circ:
                return True

    return False


def validate_circle_score(gray_img, cx, cy, r):
    """
    Validate if a candidate circle is a real verse marker by checking
    if it contains a centered contour, and return (pass, circularity).
    """
    pad = 5
    y1 = max(0, cy - r - pad)
    y2 = min(gray_img.shape[0], cy + r + pad)
    x1 = max(0, cx - r - pad)
    x2 = min(gray_img.shape[1], cx + r + pad)

    crop = gray_img[y1:y2, x1:x2]
    if crop.size == 0:
        return False, 0.0

    # Filter out low-contrast background or solid margin ornaments
    if int(np.max(crop)) - int(np.min(crop)) < 100:
        return False, 0.0

    _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    crop_h, crop_w = crop.shape[:2]
    center_x, center_y = crop_w / 2, crop_h / 2

    best_circ = 0.0
    found_valid_shape = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(cnt)
        cx_cnt = x + w / 2
        cy_cnt = y + h / 2
        dist_from_center = np.sqrt((cx_cnt - center_x)**2 + (cy_cnt - center_y)**2)

        if 20 < w < 52 and 20 < h < 52 and dist_from_center < 10:
            found_valid_shape = True
            if circularity > best_circ:
                best_circ = circularity

    return found_valid_shape, best_circ


def save_debug_markers_image(img, raw_circles, selected_markers):
    debug_img = img.copy()
    # Draw raw circles in light blue
    for c in raw_circles:
        if len(c) >= 3:
            cv2.circle(debug_img, (int(c[0]), int(c[1])), int(c[2]), (200, 200, 0), 1)
    # Draw filtered markers in red
    for i, (cx, cy, r) in enumerate(selected_markers):
        cv2.circle(debug_img, (cx, cy), r, (0, 0, 255), 2)
        cv2.circle(debug_img, (cx, cy), 2, (0, 0, 255), 3)
        cv2.putText(debug_img, f"M{i}", (cx + r + 3, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    cv2.imwrite("debug_markers.jpg", debug_img)


# ============================================================
# MARKER DETECTION (HoughCircles + line proximity filter)
# ============================================================

def detect_verse_markers(img, line_centers, expected_count=None, debug=False, page=0):
    """
    Detect verse markers using HoughCircles.
    Filter out false positives (header ornaments, border decorations)
    by requiring each detected circle to be near a detected text line.

    If expected_count is provided and the standard strict pass returns fewer markers,
    performs a relaxed pass to find weak circles and selects candidates that
    minimize the maximum line gap between markers.

    Returns list of (cx, cy, r) sorted top-to-bottom.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pass 1: Strict detection (standard parameters)
    param1_val = 80
    param2_strict = 40
    circ_strict = 0.5

    circles_strict = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=30,
        param1=param1_val, param2=param2_strict,
        minRadius=14, maxRadius=30
    )

    markers_strict = []
    raw_strict_list = []
    raw_strict_coords = []
    if circles_strict is not None:
        raw_strict = np.round(circles_strict[0]).astype(int)
        raw_strict_list = raw_strict.tolist()
        raw_strict_coords = [(int(cx), int(cy)) for cx, cy, r in raw_strict]
        for cx, cy, r in raw_strict:
            cx, cy, r = int(cx), int(cy), int(r)
            if cy < CONTENT_TOP or cy > CONTENT_BOTTOM:
                continue
            if cx < CONTENT_LEFT - 5 or cx > CONTENT_RIGHT + 5:
                continue
            if r > 23:
                continue
            near_line = any(abs(cy - yc) < BLOCK_HEIGHT * 0.8 for yc in line_centers)
            if not near_line:
                continue
            if not validate_circle(gray, cx, cy, r, thresh_circ=circ_strict):
                continue
            markers_strict.append((cx, cy, r))

    markers_strict.sort(key=lambda m: m[1])

    if expected_count is None or len(markers_strict) == expected_count:
        if debug:
            save_debug_markers_image(img, raw_strict_list, markers_strict)
        return markers_strict

    # If we got less than expected, we run relaxed Pass 2 to collect more candidates
    if len(markers_strict) < expected_count:
        print(f"    [INFO] Strict detection found {len(markers_strict)} markers, expected {expected_count}. Running relaxed Pass 2...")

        # Map strict markers to lines
        lines_with_strict = set()
        for cx, cy, r in markers_strict:
            best = min(range(len(line_centers)), key=lambda i: abs(cy - line_centers[i]))
            if abs(cy - line_centers[best]) < BLOCK_HEIGHT:
                lines_with_strict.add(best)

        # Detect likely headers (surah_header or basmallah lines) to block relaxed rescues on them
        likely_headers = set()
        for idx, yc in enumerate(line_centers):
            if idx in lines_with_strict:
                continue
            
            # strip check for Basmallah
            safe_left, safe_right = 100, 700
            y1 = max(0, yc - 20)
            y2 = min(gray.shape[0], yc + 20)
            strip = gray[y1:y2, safe_left:safe_right]
            _, binary = cv2.threshold(strip, 180, 255, cv2.THRESH_BINARY_INV)
            col_sums = np.sum(binary, axis=0)
            active_cols = np.where(col_sums > 50)[0]
            if len(active_cols) > 0:
                left = active_cols[0] + safe_left
                right = active_cols[-1] + safe_left
                width = right - left
                center = (left + right) / 2
                content_center = (safe_left + safe_right) / 2
                if width < 450 and abs(center - content_center) < 50:
                    likely_headers.add(idx)
                    if idx > 0:
                        likely_headers.add(idx - 1)

            # Row density check for surah headers
            y1 = max(0, yc - 30)
            y2 = min(gray.shape[0], yc + 30)
            strip = gray[y1:y2, CONTENT_LEFT:CONTENT_RIGHT]
            _, binary = cv2.threshold(strip, 180, 255, cv2.THRESH_BINARY_INV)
            row_densities = np.sum(binary, axis=1) / (255.0 * (CONTENT_RIGHT - CONTENT_LEFT))
            max_row_density = np.max(row_densities)
            if max_row_density > 0.80:
                likely_headers.add(idx)

        # param2=26 to detect weaker/touching circles
        circles_relaxed = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=30,
            param1=80, param2=26,
            minRadius=14, maxRadius=30
        )

        candidates = []
        raw_relaxed_list = []
        
        # Merge raw strict and relaxed circles to avoid suppression issues
        all_raw_circles = []
        filtered_strict = []
        for cx, cy, r in raw_strict_list:
            cx, cy, r = int(cx), int(cy), int(r)
            if cy < CONTENT_TOP or cy > CONTENT_BOTTOM:
                continue
            if cx < CONTENT_LEFT - 5 or cx > CONTENT_RIGHT + 5:
                continue
            if r > 23:
                continue
            near_line = any(abs(cy - yc) < BLOCK_HEIGHT * 0.8 for yc in line_centers)
            if not near_line:
                continue
            filtered_strict.append((cx, cy, r))
            all_raw_circles.append((cx, cy, r, True))
            
        if circles_relaxed is not None:
            raw_relaxed = np.round(circles_relaxed[0]).astype(int)
            raw_relaxed_list = raw_relaxed.tolist()
            for cx, cy, r in raw_relaxed:
                cx, cy, r = int(cx), int(cy), int(r)
                if cy < CONTENT_TOP or cy > CONTENT_BOTTOM:
                    continue
                if cx < CONTENT_LEFT - 5 or cx > CONTENT_RIGHT + 5:
                    continue
                if r > 23:
                    continue
                near_line = any(abs(cy - yc) < BLOCK_HEIGHT * 0.8 for yc in line_centers)
                if not near_line:
                    continue
                near_strict = any(abs(cx - scx) < 15 and abs(cy - scy) < 15 for scx, scy, sr in filtered_strict)
                if not near_strict:
                    all_raw_circles.append((cx, cy, r, False))

        for cx, cy, r, was_in_raw_strict in all_raw_circles:

            val_pass, circ_val = validate_circle_score(gray, cx, cy, r)
            
            # Block relaxed rescues on likely header and basmallah lines
            if not was_in_raw_strict:
                nearest_line_idx = min(range(len(line_centers)), key=lambda idx: abs(cy - line_centers[idx]))
                if nearest_line_idx in likely_headers:
                    continue

            # Block relaxed rescues in outer margin ornament zones to prevent false positives
            if not was_in_raw_strict and page > 0:
                is_even = (page % 2 == 0)
                if is_even and cx < 75:
                    continue
                elif not is_even and cx > 725:
                    continue

            # Determine min circularity threshold: require higher circularity in the margin ornament zones
            in_margin = False
            if page > 0:
                is_even = (page % 2 == 0)
                if is_even and cx < 130:
                    in_margin = True
                elif not is_even and cx > 670:
                    in_margin = True
            
            min_circ = 0.25 if in_margin else 0.05

            # Allow a much lower circularity (0.05) to rescue merged circles in the text area
            if val_pass and (circ_val >= min_circ or was_in_raw_strict):
                candidates.append((cx, cy, r, circ_val, was_in_raw_strict))

        # Sort candidates top-to-bottom
        candidates.sort(key=lambda m: m[1])

        # Perform minimax line-gap greedy selection to rescue markers
        selected = list(markers_strict)

        def is_selected(c):
            return any(abs(c[0] - sx) < 15 and abs(c[1] - sy) < 15 for sx, sy, sr in selected)

        remaining_candidates = [c for c in candidates if not is_selected(c)]

        while len(selected) < expected_count and remaining_candidates:
            best_c = None
            best_score = -9999.0

            for c in remaining_candidates:
                cx, cy, cr, circ, was_in_raw_strict = c
                c_line = min(range(len(line_centers)), key=lambda idx: abs(cy - line_centers[idx]))

                temp_selected = selected + [(cx, cy, cr)]
                sel_lines = sorted(list(set(
                    min(range(len(line_centers)), key=lambda idx: abs(sy - line_centers[idx]))
                    for sx, sy, sr in temp_selected
                )))

                gaps = []
                if sel_lines:
                    gaps.append(sel_lines[0])
                    for idx in range(1, len(sel_lines)):
                        gaps.append(sel_lines[idx] - sel_lines[idx-1])
                    gaps.append((len(line_centers) - 1) - sel_lines[-1])
                max_gap = max(gaps) if gaps else len(line_centers)

                # Penalize if it shares a line index with an already selected marker
                shares_line = any(
                    min(range(len(line_centers)), key=lambda idx: abs(sy - line_centers[idx])) == c_line
                    for sx, sy, sr in selected
                )

                shares_penalty = 10.0 if shares_line else 0.0
                
                # Prioritize raw strict circles heavily over relaxed ones
                raw_strict_bonus = 100.0 if was_in_raw_strict else 0.0
                
                score = (100 - max_gap) - shares_penalty + circ + raw_strict_bonus

                if score > best_score:
                    best_score = score
                    best_c = c

            if best_c is not None:
                cx, cy, cr, circ, was_in_raw_strict = best_c
                selected.append((cx, cy, cr))
                remaining_candidates = [c for c in remaining_candidates if not is_selected(c)]
                print(f"      [RESCUE] Rescued relaxed candidate at ({cx}, {cy}) r={cr} (circ={circ:.2f}, was_in_raw_strict={was_in_raw_strict})")
            else:
                break

        selected.sort(key=lambda m: m[1])

        if debug:
            all_raw = raw_strict_list + raw_relaxed_list
            save_debug_markers_image(img, all_raw, selected)

        return selected

    # If we got more than expected, prune the ones with the lowest circularity
    if len(markers_strict) > expected_count:
        print(f"    [INFO] Strict detection found {len(markers_strict)} markers, expected {expected_count}. Pruning...")
        scored_strict = []
        for cx, cy, r in markers_strict:
            _, circ_val = validate_circle_score(gray, cx, cy, r)
            scored_strict.append((cx, cy, r, circ_val))

        scored_strict.sort(key=lambda x: -x[3])
        pruned = [(cx, cy, r) for cx, cy, r, circ in scored_strict[:expected_count]]
        pruned.sort(key=lambda m: m[1])
        return pruned


# ============================================================
# MARKER → LINE ASSIGNMENT
# ============================================================

def assign_markers_to_lines(markers, line_centers):
    """
    Assign each marker to its nearest text line.
    Returns dict: line_index -> list of (cx, r) sorted right-to-left.
    """
    line_markers = {}

    for cx, cy, r in markers:
        best = min(range(len(line_centers)), key=lambda i: abs(cy - line_centers[i]))
        if abs(cy - line_centers[best]) < BLOCK_HEIGHT:
            line_markers.setdefault(best, []).append((cx, r))

    for idx in line_markers:
        line_markers[idx].sort(key=lambda m: -m[0])

    return line_markers


# ============================================================
# LINE CLASSIFICATION (Headers and Basmallah)
# ============================================================

def classify_lines(img, line_centers, line_markers, start_ayat=1, expected_surahs=None):
    """
    Classify each text line as 'surah_header', 'basmallah', or 'normal_text'.
    Modifies line_markers in-place to reassign markers falling on header/basmallah lines.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    classifications = ['normal_text'] * len(line_centers)
    basmallah_indices = []

    # 1. Detect Basmallah lines
    for idx, yc in enumerate(line_centers):
        # A line with multiple verse markers can NEVER be a Basmallah
        if idx in line_markers and len(line_markers[idx]) > 1:
            continue

        # Get active text bounds using safe inner margins to ignore side ornaments
        safe_left, safe_right = 100, 700
        y1 = max(0, yc - 20)
        y2 = min(h, yc + 20)
        strip = gray[y1:y2, safe_left:safe_right]
        _, binary = cv2.threshold(strip, 180, 255, cv2.THRESH_BINARY_INV)
        col_sums = np.sum(binary, axis=0)
        active_cols = np.where(col_sums > 50)[0]

        if len(active_cols) > 0:
            left = active_cols[0] + safe_left
            right = active_cols[-1] + safe_left
            width = right - left
            center = (left + right) / 2
            content_center = (safe_left + safe_right) / 2

            # Basmallah is centered and narrow (width < 450)
            is_basmallah = (width < 450) and (abs(center - content_center) < 50)
            if is_basmallah:
                basmallah_indices.append(idx)
                classifications[idx] = 'basmallah'

    # 2. Detect Surah Headers preceding Basmallah
    for b_idx in basmallah_indices:
        for p_idx in range(b_idx - 1, -1, -1):
            if abs(line_centers[p_idx] - line_centers[b_idx]) > 160:
                break
            classifications[p_idx] = 'surah_header'
            break

    # Dynamic Marker Reassignment Helper
    def reassign_bad_markers():
        for idx in list(line_markers.keys()):
            if classifications[idx] in ['surah_header', 'basmallah']:
                bad_markers = line_markers.pop(idx)
                for mx, mr in bad_markers:
                    best_normal = min(
                        [n_idx for n_idx, cls in enumerate(classifications) if cls == 'normal_text'],
                        key=lambda n_idx: abs(n_idx - idx),
                        default=None
                    )
                    if best_normal is not None:
                        line_markers.setdefault(best_normal, []).append((mx, mr))
                        line_markers[best_normal].sort(key=lambda m: m[0], reverse=True)
                        print(f"      [REASSIGN] Moved marker at x={mx} from Line {idx} ({classifications[idx]}) to Line {best_normal} (normal_text)")



    # 3. Detect Surah Headers without Basmallah (At-Taubah style) using density peak check
    for idx, yc in enumerate(line_centers):
        if classifications[idx] != 'normal_text':
            continue
        if idx in line_markers:
            continue

        # Check max row density in a wider window around the center
        y1 = max(0, yc - 30)
        y2 = min(h, yc + 30)
        strip = gray[y1:y2, CONTENT_LEFT:CONTENT_RIGHT]
        _, binary = cv2.threshold(strip, 180, 255, cv2.THRESH_BINARY_INV)
        row_densities = np.sum(binary, axis=1) / (255.0 * (CONTENT_RIGHT - CONTENT_LEFT))
        max_row_density = np.max(row_densities)

        if max_row_density > 0.80:
            classifications[idx] = 'surah_header'

    # 4. Post-process: any line immediately following a surah_header
    # that has no markers MUST be a basmallah (e.g. wide Basmallah layouts).
    for idx in range(len(classifications) - 1):
        if classifications[idx] == 'surah_header':
            next_idx = idx + 1
            if next_idx not in line_markers:
                classifications[next_idx] = 'basmallah'

    # 5. Metadata-based validation
    if expected_surahs:
        if len(expected_surahs) == 1:
            if start_ayat > 1:
                # No surah headers or basmallahs should be present anywhere on the page
                classifications = ['normal_text'] * len(line_centers)
            else:
                # Surah starts at the top (start_ayat == 1). We only allow headers/basmallahs at the very top (idx <= 2).
                for idx in range(len(classifications)):
                    if idx > 2:
                        classifications[idx] = 'normal_text'
        else:
            # The number of surah headers cannot exceed the number of expected surahs on the page
            max_headers = len(expected_surahs)
            header_indices = [idx for idx, cls in enumerate(classifications) if cls == 'surah_header']
            if len(header_indices) > max_headers:
                for idx in header_indices[max_headers:]:
                    classifications[idx] = 'normal_text'
                    if idx + 1 < len(classifications) and classifications[idx + 1] == 'basmallah':
                        classifications[idx + 1] = 'normal_text'

    # Final reassignment cleanup in case step 3/4/5 changed classifications
    reassign_bad_markers()

    return classifications


# ============================================================
# HIGHLIGHT GENERATION
# ============================================================

# Standard number of verses in each of the 114 surahs (1-indexed)
SURAH_VERSES = [
    0, # placeholder for index 0
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, # 1-10
    123, 111, 43, 52, 99, 128, 111, 110, 98, 135, # 11-20
    112, 78, 118, 64, 77, 227, 93, 88, 69, 60, # 21-30
    34, 30, 73, 54, 45, 83, 182, 88, 75, 85, # 31-40
    54, 53, 89, 59, 37, 35, 38, 29, 18, 45, # 41-50
    60, 49, 62, 55, 78, 96, 29, 22, 24, 13, # 51-60
    14, 11, 11, 18, 12, 12, 30, 52, 52, 44, # 61-70
    28, 28, 20, 56, 40, 31, 50, 40, 46, 42, # 71-80
    29, 19, 36, 25, 22, 17, 19, 26, 30, 20, # 81-90
    15, 21, 11, 8, 8, 19, 5, 8, 8, 11, # 91-100
    11, 8, 3, 9, 5, 4, 7, 3, 6, 3, # 101-110
    5, 4, 5, 6 # 111-114
]

def generate_highlights(line_centers, line_markers, start_surah, start_ayat, classifications, adjust_y=0, expected_surahs=None):
    """
    Generate highlight regions per verse. Handles surah transitions when headers are hit.
    """
    highlights = {}
    current_surah = start_surah
    current_ayat = start_ayat
    has_started_verses = False

    for line_idx, yc in enumerate(line_centers):
        cls = classifications[line_idx]

        if cls == 'surah_header':
            if has_started_verses:
                next_surah = current_surah + 1
                if expected_surahs and next_surah not in expected_surahs:
                    # Treat this line as normal text instead of transitioning
                    cls = 'normal_text'
                else:
                    current_surah = next_surah
                    current_ayat = 1
                    has_started_verses = False
                    continue

        if cls == 'basmallah':
            continue

        # Normal text line
        has_started_verses = True
        block_top = int(yc - BLOCK_HEIGHT / 2) + adjust_y
        markers = line_markers.get(line_idx, [])

        # Check if this is the last text line before a new surah header or end of page
        is_end_of_surah_line = False
        if line_idx == len(line_centers) - 1:
            is_end_of_surah_line = True
        elif classifications[line_idx + 1] in ['surah_header', 'basmallah']:
            is_end_of_surah_line = True

        max_verses = SURAH_VERSES[current_surah] if current_surah < len(SURAH_VERSES) else 999

        if not markers:
            if current_ayat <= max_verses:
                _add(highlights, current_surah, current_ayat, CONTENT_LEFT, block_top,
                     CONTENT_RIGHT - CONTENT_LEFT, BLOCK_HEIGHT)
        else:
            right_edge = CONTENT_RIGHT
            for mcx, mr in markers:
                m_right = mcx + mr + 3
                m_left = mcx - mr - 3

                if current_ayat <= max_verses:
                    if right_edge > m_right + 5:
                        _add(highlights, current_surah, current_ayat, m_right, block_top,
                             right_edge - m_right, BLOCK_HEIGHT)

                current_ayat += 1
                right_edge = m_left

            # If this is the last line of a surah, do not add the empty space to the left of the last marker
            if right_edge > CONTENT_LEFT + 5:
                if not is_end_of_surah_line:
                    if current_ayat <= max_verses:
                        _add(highlights, current_surah, current_ayat, CONTENT_LEFT, block_top,
                             right_edge - CONTENT_LEFT, BLOCK_HEIGHT)

    return highlights


def _add(highlights, surah, ayat, left, top, width, height):
    key = (surah, ayat)
    highlights.setdefault(key, []).append({
        'left': int(left), 'top': int(top),
        'width': int(width), 'height': int(height)
    })


# ============================================================
# VISUALIZATION
# ============================================================

def draw_highlights(img, highlights):
    annotated = img.copy()
    sorted_keys = sorted(highlights.keys(), key=lambda k: (k[0], k[1]))

    for idx, (s, a) in enumerate(sorted_keys):
        color = VERSE_COLORS[idx % len(VERSE_COLORS)]
        rects = highlights[(s, a)]

        for hi, r in enumerate(rects):
            x, y, rw, rh = r['left'], r['top'], r['width'], r['height']

            overlay = annotated.copy()
            cv2.rectangle(overlay, (x, y), (x + rw, y + rh), color, -1)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
            cv2.rectangle(annotated, (x, y), (x + rw, y + rh), color, 2)

            if hi == 0:
                label = f"{s}:{a}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.5, 2)
                cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 10, y), color, -1)
                cv2.putText(annotated, label, (x + 5, y - 4), font, 0.5, (0, 0, 0), 2)

    return annotated


def load_image(path):
    """
    Load image from path, supporting transparent WebP/PNG by blending
    the alpha channel over a white background.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha = a.astype(float) / 255.0
        bg = np.ones_like(a) * 255
        b = (b.astype(float) * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        g = (g.astype(float) * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        r = (r.astype(float) * alpha + bg * (1.0 - alpha)).astype(np.uint8)
        img = cv2.merge([b, g, r])
    return img


# ============================================================
# SPECIAL ILLUMINATION PAGES PROCESSING (Page 1 & 2)
# ============================================================

def process_special_page(img, page, surah, start_ayat):
    """
    Specialized pipeline for Page 1 (Al-Fatihah) and Page 2 (Al-Baqarah start)
    which are full-decoration illumination pages with thick borders.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Define bounds
    c_left, c_right = 150, 650
    if page == 1:
        c_top, c_bottom = 380, 920
        max_ayat = 7
    else:
        c_top, c_bottom = 380, 980
        max_ayat = 5
        
    # 2. Line detection (projection-based in safe center area)
    content = gray[c_top:c_bottom, c_left:c_right]
    _, binary = cv2.threshold(content, 180, 255, cv2.THRESH_BINARY_INV)
    projection = np.sum(binary, axis=1).astype(float)
    kernel = np.ones(7) / 7
    projection = np.convolve(projection, kernel, mode='same')
    pmax = np.max(projection)
    if pmax > 0:
        projection /= pmax

    min_height = 0.30
    min_distance = 70
    peaks = []
    for i in range(1, len(projection) - 1):
        if (projection[i] > min_height and
            projection[i] >= projection[i - 1] and
            projection[i] >= projection[i + 1]):
            peaks.append(i)

    filtered = []
    for p in peaks:
        if filtered:
            prev = filtered[-1]
            if p - prev < min_distance:
                valley_val = np.min(projection[prev:p+1])
                if valley_val < 0.10:
                    filtered.append(p)
                else:
                    if projection[p] > projection[prev]:
                        filtered[-1] = p
            else:
                filtered.append(p)
        else:
            filtered.append(p)

    line_centers = [p + c_top for p in filtered]
    print(f"    Found {len(line_centers)} text lines on illumination page {page}")
    for idx, yc in enumerate(line_centers):
        print(f"      Line {idx:2d}: y_center={yc}")
    
    # 3. Detect verse markers using more sensitive Hough parameter (param2=20)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=70, param2=20, minRadius=12, maxRadius=24
    )
    
    markers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for cx, cy, r in circles[0]:
            # Broaden range slightly to catch markers next to lines
            if c_left < cx < c_right and c_top - 50 < cy < c_bottom + 50:
                if validate_circle(gray, cx, cy, r):
                    markers.append((int(cx), int(cy), int(r)))
                    
    # Deduplicate close circles
    unique_markers = []
    for m in markers:
        too_close = False
        for um in unique_markers:
            dist = np.sqrt((m[0]-um[0])**2 + (m[1]-um[1])**2)
            if dist < 20:
                too_close = True
                break
        if not too_close:
            unique_markers.append(m)
            
    print(f"    Found {len(unique_markers)} verse markers on illumination page {page}")
    for idx, m in enumerate(unique_markers):
        print(f"      Marker {idx}: ({m[0]}, {m[1]}) r={m[2]}")
    
    # 4. Group markers by closest line center to enable reading order sorting
    line_groups = {idx: [] for idx in range(len(line_centers))}
    for m in unique_markers:
        cx, cy, r = m
        closest_line = None
        min_d = 9999
        for idx, yc in enumerate(line_centers):
            dist = abs(cy - yc)
            if dist < min_d:
                min_d = dist
                closest_line = idx
        if closest_line is not None:
            line_groups[closest_line].append(m)
            
    # Sort markers within each line from right-to-left (descending x)
    sorted_markers = []
    for idx in sorted(line_groups.keys()):
        line_groups[idx].sort(key=lambda m: m[0], reverse=True)
        sorted_markers.extend(line_groups[idx])
        
    # 5. Static map assignment to lines
    line_markers = {idx: [] for idx in range(len(line_centers))}
    if page == 1 and len(sorted_markers) == 7:
        line_markers[0].append((sorted_markers[0][0], sorted_markers[0][2]))
        line_markers[1].append((sorted_markers[1][0], sorted_markers[1][2]))
        line_markers[2].append((sorted_markers[2][0], sorted_markers[2][2]))
        line_markers[2].append((sorted_markers[3][0], sorted_markers[3][2]))
        line_markers[3].append((sorted_markers[4][0], sorted_markers[4][2]))
        line_markers[4].append((sorted_markers[5][0], sorted_markers[5][2]))
        line_markers[6].append((sorted_markers[6][0], sorted_markers[6][2]))
    elif page == 2 and len(sorted_markers) == 5:
        line_markers[1].append((sorted_markers[0][0], sorted_markers[0][2]))
        line_markers[2].append((sorted_markers[1][0], sorted_markers[1][2]))
        line_markers[3].append((sorted_markers[2][0], sorted_markers[2][2]))
        line_markers[5].append((sorted_markers[3][0], sorted_markers[3][2]))
        line_markers[6].append((sorted_markers[4][0], sorted_markers[4][2]))
    else:
        # Fallback to closest line distance assignment if markers count doesn't match
        print("    [Warning] Markers count does not match expected (7 for page 1, 5 for page 2). Using fallback assignment.")
        for idx, ms in line_groups.items():
            for m in ms:
                line_markers[idx].append((m[0], m[2]))
                
    # 6. Build highlight blocks
    highlights = {}
    active_surah = surah
    active_ayat = start_ayat
    
    for l_idx, yc in enumerate(line_centers):
        if page == 2 and l_idx == 0:
            continue
        markers_on_line = line_markers.get(l_idx, [])
        markers_on_line.sort(key=lambda m: m[0], reverse=True)
        
        last_x = c_right
        
        for mx, mr in markers_on_line:
            if active_ayat > max_ayat:
                break
                
            r_left = mx + mr
            r_right = last_x
            r_left = max(c_left, r_left)
            r_right = min(c_right, r_right)
            
            if r_right > r_left:
                rect = {
                    'left': int(r_left),
                    'top': int(yc - 34),
                    'width': int(r_right - r_left),
                    'height': int(68)
                }
                highlights.setdefault((active_surah, active_ayat), []).append(rect)
                
            active_ayat += 1
            last_x = mx - mr
            
        if active_ayat <= max_ayat and last_x > c_left:
            r_left = c_left
            r_right = last_x
            if r_right > r_left:
                rect = {
                    'left': int(r_left),
                    'top': int(yc - 34),
                    'width': int(r_right - r_left),
                    'height': int(68)
                }
                highlights.setdefault((active_surah, active_ayat), []).append(rect)
                
    return highlights


# ============================================================
# MAIN
# ============================================================

def main():
    global BLOCK_HEIGHT, CONTENT_LEFT, CONTENT_RIGHT

    parser = argparse.ArgumentParser(description="Quran Verse Auto-Marker Tool")
    parser.add_argument('image', help="Path to Quran page image")
    parser.add_argument('--surah', type=int, required=True, help="Surah number")
    parser.add_argument('--start-ayat', type=int, required=True, help="First ayat on this page")
    parser.add_argument('--end-ayat', type=int, default=None, help="Expected last ayat (verification)")
    parser.add_argument('--expected-count', type=int, default=None, help="Expected number of verse markers on the page")
    parser.add_argument('--output', type=str, default=None, help="Output JSON path")
    parser.add_argument('--block-height', type=int, default=BLOCK_HEIGHT,
                        help=f"Block height (default: {BLOCK_HEIGHT})")
    parser.add_argument('--adjust-y', type=int, default=0,
                        help="Adjustment/offset for Y coordinates (can be positive or negative)")
    parser.add_argument('--debug', action='store_true', help="Save debug images")
    parser.add_argument('--expected-surahs', type=str, default=None, help="Comma-separated list of expected surah numbers on the page")
    args = parser.parse_args()

    expected_surahs = None
    if args.expected_surahs:
        try:
            expected_surahs = {int(x) for x in args.expected_surahs.split(",")}
        except Exception:
            pass

    BLOCK_HEIGHT = args.block_height

    # Extract page number to determine odd/even layout
    page = 0
    stem = Path(args.image).stem
    digits = ''.join(c for c in stem if c.isdigit())
    if digits:
        page = int(digits)

    if page > 0 and page % 2 == 1:
        # Odd page (Right side page): Outer margin is on the right, containing side ornaments.
        # Restrict CONTENT_RIGHT to 752 to prevent overlap with right ornaments.
        CONTENT_LEFT = 38
        CONTENT_RIGHT = 752
        print(f"    [Odd Page Layout] Setting CONTENT_LEFT = 38, CONTENT_RIGHT = 752 to prevent overlap with right margin ornaments.")
    else:
        # Even page (Left side page): Outer margin is on the left, right side is the inner margin (no ornaments).
        # We can safely use 772 for a wider highlight matching the margin.
        CONTENT_LEFT = 38
        CONTENT_RIGHT = 772
        print(f"    [Even Page Layout] Setting CONTENT_LEFT = 38, CONTENT_RIGHT = 772 for wider content bounds.")

    if not os.path.exists(args.image):
        print(f"Error: '{args.image}' not found")
        sys.exit(1)

    img = load_image(args.image)
    if img is None:
        print(f"Error: '{args.image}' cannot be loaded")
        sys.exit(1)

    h, w = img.shape[:2]
    if w != 800 or h != 1293:
        print(f"    ⚠️ Resizing image from {w}x{h} to 800x1293 to match standard coordinates system")
        img = cv2.resize(img, (800, 1293), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    print(f"Image: {w}x{h} | Block height: {BLOCK_HEIGHT}px")

    # Determine if this is an illumination page
    actual_page = page
    if page == 0:
        if args.surah == 1:
            actual_page = 1
        elif args.surah == 2 and args.start_ayat == 1:
            actual_page = 2

    if actual_page in [1, 2]:
        print(f"\n[1] Processing illumination page {actual_page} automatically...")
        highlights = process_special_page(img, actual_page, args.surah, args.start_ayat)
    else:
        # 1. Detect text lines FIRST (needed for marker filtering)
        print("\n[1] Detecting text lines...")
        line_centers = detect_text_lines(img, debug=args.debug, page=page)
        print(f"    Found {len(line_centers)} lines")
        for i, yc in enumerate(line_centers):
            print(f"      Line {i:2d}: y_center={yc}")

        # 2. Detect verse markers (filtered by text line proximity)
        print("\n[2] Detecting verse markers...")
        markers = detect_verse_markers(img, line_centers, expected_count=args.expected_count, debug=args.debug, page=page)
        print(f"    Found {len(markers)} markers (after filtering)")
        for i, (cx, cy, r) in enumerate(markers):
            print(f"      [{i}] ({cx}, {cy}) r={r}")

        # 3. Assign markers to lines
        print("\n[3] Assigning markers to lines...")
        line_markers = assign_markers_to_lines(markers, line_centers)
        for li, md in sorted(line_markers.items()):
            xs = ', '.join(f"x={cx}" for cx, r in md)
            print(f"      Line {li}: {xs}")

        # 4. Classify lines (Headers / Basmallah / Normal Text)
        print("\n[4] Classifying text lines...")
        classifications = classify_lines(
            img, line_centers, line_markers, 
            start_ayat=args.start_ayat, 
            expected_surahs=expected_surahs
        )
        for i, cls in enumerate(classifications):
            print(f"      Line {i:2d} (y={line_centers[i]:4d}): {cls}")



        # 5. Generate highlights
        print(f"\n[5] Generating highlights...")
        highlights = generate_highlights(
            line_centers, line_markers, args.surah, args.start_ayat,
            classifications, adjust_y=args.adjust_y,
            expected_surahs=expected_surahs
        )

    total = sum(len(v) for v in highlights.values())
    sorted_keys = sorted(highlights.keys(), key=lambda k: (k[0], k[1]))
    if sorted_keys:
        start_key = sorted_keys[0]
        end_key = sorted_keys[-1]
        print(f"    {len(highlights)} verses, {total} blocks, range: {start_key[0]}:{start_key[1]} → {end_key[0]}:{end_key[1]}")
    else:
        print("    No verses generated")

    # 6. Save JSON
    stem = Path(args.image).stem
    out_json = args.output or str(Path(args.image).parent / f"{stem}_highlights.json")
    page = int(''.join(c for c in stem if c.isdigit()) or '0')

    verses_list = []
    for s, a in sorted_keys:
        verses_list.append({
            'surah': s,
            'ayat': a,
            'highlights': highlights[(s, a)]
        })

    data = {
        'page': page,
        'image': os.path.basename(args.image),
        'image_width': w, 'image_height': h,
        'surah': args.surah,
        'verses': verses_list
    }
    with open(out_json, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON: {out_json}")

    # 7. Save annotated image
    out_img = str(Path(args.image).parent / f"{stem}_highlighted.jpg")
    cv2.imwrite(out_img, draw_highlights(img, highlights),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✅ Image: {out_img}")

    # Summary
    print(f"\n{'='*50}")
    for s, a in sorted_keys:
        n = len(highlights[(s, a)])
        print(f"  Surah {s:3d} Ayat {a:3d} → {n} block{'s' if n > 1 else ' '}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
