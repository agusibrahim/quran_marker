#!/usr/bin/env python3
"""
Analyze the actual verse marker more precisely.
Crop one known marker and understand its properties.
"""
import cv2
import numpy as np

img = cv2.imread("QK_004.jpg")
h, w = img.shape[:2]

# From the initial HoughCircles analysis, these were the GOOD detections
# (circles detected that actually ARE verse markers):
# center=(500, 1119) r=20  -> this is marker ٢٣ or ٢٤
# center=(599, 645) r=20   -> this is marker ٢٠
# center=(133, 172) r=20   -> this is marker ١٧
# center=(625, 957) r=18   -> this is marker ٢٣
# center=(307, 406) r=17   -> this is marker ١٩
# center=(299, 727) r=24   -> this is marker ٢١ (maybe ٢٠)
# center=(407, 253) r=15   -> this is marker ١٨

# Let's crop these known markers and analyze their properties
known_markers = [
    (133, 172, 20, "17"),
    (407, 253, 18, "18"),
    (307, 406, 20, "19"),
    (599, 645, 22, "20"),
    (299, 727, 24, "21"),
    (500, 958, 20, "22"),  # adjusted
    (625, 1045, 20, "23"),
    (80, 1114, 22, "24"),   # last line
]

pad = 35
for cx, cy, r, name in known_markers:
    crop = img[max(0,cy-pad):min(h,cy+pad), max(0,cx-pad):min(w,cx+pad)]
    cv2.imwrite(f"debug_marker_{name}.jpg", crop)
    
    # Analyze HSV in this region
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Compute mean and std of H, S, V
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (pad, pad), r+5, 255, -1)
    
    h_vals = hsv_crop[:,:,0][mask > 0]
    s_vals = hsv_crop[:,:,1][mask > 0]
    v_vals = hsv_crop[:,:,2][mask > 0]
    
    print(f"Marker {name} at ({cx},{cy}) r={r}:")
    print(f"  H: mean={h_vals.mean():.1f} std={h_vals.std():.1f} range=[{h_vals.min()}-{h_vals.max()}]")
    print(f"  S: mean={s_vals.mean():.1f} std={s_vals.std():.1f} range=[{s_vals.min()}-{s_vals.max()}]")
    print(f"  V: mean={v_vals.mean():.1f} std={v_vals.std():.1f} range=[{v_vals.min()}-{v_vals.max()}]")

# Also, let's try a different approach: template matching
# Use one marker as template and find all similar ones
print("\n--- Template matching approach ---")

# Crop marker 20 as template (good size, clear)
tcx, tcy, tr = 599, 645, 22
tpad = 28
template = img[tcy-tpad:tcy+tpad, tcx-tpad:tcx+tpad]
cv2.imwrite("debug_template.jpg", template)

# Convert both to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Multi-scale template matching
best_results = []
for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
    tw = int(template_gray.shape[1] * scale)
    th = int(template_gray.shape[0] * scale)
    if tw < 10 or th < 10:
        continue
    
    scaled = cv2.resize(template_gray, (tw, th))
    result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
    
    # Find all matches above threshold
    threshold = 0.45
    locations = np.where(result >= threshold)
    
    for pt_y, pt_x in zip(*locations):
        cx = pt_x + tw // 2
        cy = pt_y + th // 2
        score = result[pt_y, pt_x]
        best_results.append((cx, cy, score, scale))

# Non-maximum suppression
best_results.sort(key=lambda r: -r[2])
final_markers = []
for cx, cy, score, scale in best_results:
    too_close = False
    for fcx, fcy, _, _ in final_markers:
        if abs(cx - fcx) < 30 and abs(cy - fcy) < 30:
            too_close = True
            break
    if not too_close:
        final_markers.append((cx, cy, score, scale))

# Filter by content area
content_markers = [(cx, cy, score, scale) for cx, cy, score, scale in final_markers 
                   if 50 < cy < 1130 and 50 < cx < 760]
content_markers.sort(key=lambda m: m[1])

print(f"Template matches (threshold=0.45): {len(content_markers)}")
for i, (cx, cy, score, scale) in enumerate(content_markers):
    print(f"  [{i}] center=({cx}, {cy}) score={score:.3f} scale={scale}")

# Draw template matches
tmatch_img = img.copy()
for cx, cy, score, scale in content_markers:
    r = int(22 * scale)
    cv2.circle(tmatch_img, (cx, cy), r, (0, 0, 255), 2)
    cv2.putText(tmatch_img, f"{score:.2f}", (cx+r+2, cy),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
cv2.imwrite("debug_template_match.jpg", tmatch_img)
print("Saved debug_template_match.jpg")
