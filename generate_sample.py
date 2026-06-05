#!/usr/bin/env python3
"""
Generate sample JSON output and annotated image for QK_004.jpg (Al-Baqarah page 4, ayat 17-24).
This demonstrates the expected output format and renders highlight blocks per verse.

Coordinates have been carefully adjusted to match the actual verse positions.
"""

import cv2
import numpy as np
import json
import os

IMAGE_PATH = "QK_004.jpg"
OUTPUT_JSON = "QK_004_highlights.json"
OUTPUT_IMAGE = "QK_004_highlighted.jpg"

# Image is 800 x 1293
# Content area approximately: left=58, right=752, top=52, line_height ~72px
# Lines start at approximately y=52 and increment by ~72px each
# There are ~15 text lines visible

# Line Y positions (top of each text line)
# Measured from the image:
L = [52, 122, 192, 262, 332, 402, 472, 542, 612, 682, 752, 822, 892, 962, 1042]
LH = 68  # line height for each block
LEFT = 58  # left margin of text content
RIGHT = 752  # right edge of text content
W_FULL = RIGHT - LEFT  # full line width

# Verse markers (circled numbers) positions observed in the image:
# ١٧ at line 2, roughly x=240
# ١٨ at line 3, roughly x=425
# ١٩ at line 5, roughly x=375
# ٢٠ at line 8, roughly x=555
# ٢١ at line 9, roughly x=405
# ٢٢ at line 12, roughly x=580
# ٢٣ at line 14, roughly x=365
# ٢٤ at end, line 15, roughly x=90

VERSE_DATA = [
    {
        "surah": 2,
        "ayat": 17,
        "highlights": [
            # Line 1 (full): مثلهم كمثل الذي استوقد نارا فلما اضاءت ما حوله
            {"left": LEFT, "top": L[0], "width": W_FULL, "height": LH},
            # Line 2 (from right to marker ١٧ at ~x=240):
            {"left": 240, "top": L[1], "width": RIGHT - 240, "height": LH},
        ]
    },
    {
        "surah": 2,
        "ayat": 18,
        "highlights": [
            # Line 2 (from left to marker ١٧, "صم"):
            {"left": LEFT, "top": L[1], "width": 182, "height": LH},
            # Line 3 (from right to marker ١٨ at ~x=425):
            {"left": 425, "top": L[2], "width": RIGHT - 425, "height": LH},
        ]
    },
    {
        "surah": 2,
        "ayat": 19,
        "highlights": [
            # Line 3 (from marker ١٨ to left): او كصيب من السماء فيه
            {"left": LEFT, "top": L[2], "width": 367, "height": LH},
            # Line 4 (full): ظلمت ورعد وبرق...
            {"left": LEFT, "top": L[3], "width": W_FULL, "height": LH},
            # Line 5 (from right to marker ١٩ at ~x=375):
            {"left": 375, "top": L[4], "width": RIGHT - 375, "height": LH},
        ]
    },
    {
        "surah": 2,
        "ayat": 20,
        "highlights": [
            # Line 5 (from marker ١٩ to left): يكاد البرق يخطف
            {"left": LEFT, "top": L[4], "width": 317, "height": LH},
            # Line 6 (full): ابصارهم كلما اضاء...
            {"left": LEFT, "top": L[5], "width": W_FULL, "height": LH},
            # Line 7 (full): ولو شاء الله...
            {"left": LEFT, "top": L[6], "width": W_FULL, "height": LH},
            # Line 8 (from right to marker ٢٠ at ~x=555): شيء قدير
            {"left": 555, "top": L[7], "width": RIGHT - 555, "height": LH},
        ]
    },
    {
        "surah": 2,
        "ayat": 21,
        "highlights": [
            # Line 8 (from marker ٢٠ to left): يا ايها الناس اعبدوا ربكم الذي خلقكم
            {"left": LEFT, "top": L[7], "width": 497, "height": LH},
            # Line 9 (from right to marker ٢١ at ~x=405):
            {"left": 405, "top": L[8], "width": RIGHT - 405, "height": LH},
        ]
    },
    {
        "surah": 2,
        "ayat": 22,
        "highlights": [
            # Line 9 (from marker ٢١ to left): الذي جعل لكم
            {"left": LEFT, "top": L[8], "width": 347, "height": LH},
            # Line 10 (full):
            {"left": LEFT, "top": L[9], "width": W_FULL, "height": LH},
            # Line 11 (full):
            {"left": LEFT, "top": L[10], "width": W_FULL, "height": LH},
            # Line 12 (from right to marker ٢٢ at ~x=580): تعلمون
            {"left": 580, "top": L[11], "width": RIGHT - 580, "height": LH},
        ]
    },
    {
        "surah": 2,
        "ayat": 23,
        "highlights": [
            # Line 12 (from marker ٢٢ to left): وان كنتم في ريب...
            {"left": LEFT, "top": L[11], "width": 522, "height": LH},
            # Line 13 (full):
            {"left": LEFT, "top": L[12], "width": W_FULL, "height": LH},
            # Line 14 (from right to marker ٢٣ at ~x=365):
            {"left": 365, "top": L[13], "width": RIGHT - 365, "height": LH},
        ]
    },
    {
        "surah": 2,
        "ayat": 24,
        "highlights": [
            # Line 14 (from marker ٢٣ to left): فان لم تفعلوا ولن تفعلوا فاتقوا
            {"left": LEFT, "top": L[13], "width": 307, "height": LH},
            # Line 15 (full, last line): النار التي وقودها الناس والحجارة اعدت للكفرين
            {"left": LEFT, "top": L[14], "width": W_FULL, "height": LH},
        ]
    },
]

# Color palette (BGR for OpenCV)
COLORS = [
    (255, 180, 50),    # Cyan-ish
    (50, 220, 130),    # Green
    (80, 130, 255),    # Red-orange
    (220, 100, 255),   # Pink
    (50, 255, 255),    # Yellow
    (255, 100, 100),   # Blue
    (100, 255, 200),   # Mint
    (200, 50, 255),    # Magenta
]


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Cannot load {IMAGE_PATH}")
        return

    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")

    # ---- Generate JSON ----
    json_data = {
        "page": 4,
        "image": IMAGE_PATH,
        "image_width": w,
        "image_height": h,
        "surah": 2,
        "verses": VERSE_DATA
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON saved: {OUTPUT_JSON}")
    print(f"   {len(VERSE_DATA)} verses, {sum(len(v['highlights']) for v in VERSE_DATA)} highlight blocks")

    # ---- Generate annotated image ----
    annotated = img.copy()

    for idx, verse in enumerate(VERSE_DATA):
        color = COLORS[idx % len(COLORS)]
        ayat = verse['ayat']

        for hi, hl in enumerate(verse['highlights']):
            x, y, bw, bh = hl['left'], hl['top'], hl['width'], hl['height']

            # Draw semi-transparent fill
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, -1)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

            # Draw border
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

            # Label on first block of each verse
            if hi == 0:
                label = f"{verse['surah']}:{ayat}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                lx, ly = x, y - 2
                cv2.rectangle(annotated, (lx, ly - th - 6), (lx + tw + 10, ly + 2), color, -1)
                cv2.putText(annotated, label, (lx + 5, ly - 3), font, font_scale, (0, 0, 0), thickness)

    cv2.imwrite(OUTPUT_IMAGE, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✅ Annotated image saved: {OUTPUT_IMAGE}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  JSON Output Summary")
    print(f"{'='*50}")
    for verse in VERSE_DATA:
        n = len(verse['highlights'])
        print(f"  Surah {verse['surah']} Ayat {verse['ayat']:2d} → {n} block{'s' if n > 1 else ' '}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
