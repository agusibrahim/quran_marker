#!/usr/bin/env python3
"""
Quran Verse Highlight Marker Tool
===================================
Interactive tool to mark/highlight individual verses (ayat) on Quran page images.
Outputs JSON with coordinates for each verse's highlight regions.

Usage:
    python marker.py QK_004.jpg --surah 2 --start-ayat 17
    python marker.py QK_004.jpg --surah 2 --start-ayat 17 --output custom_output.json

Controls:
    Mouse drag  : Draw highlight rectangle
    N           : Next ayat
    P           : Previous ayat
    U           : Undo last rectangle
    R           : Reset current ayat
    S           : Save to JSON
    Q / ESC     : Quit (auto-save)
"""

import cv2
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path


# Fixed block height for each highlight (pixels)
BLOCK_HEIGHT = 68

# Color palette for verse highlights (BGR format for OpenCV)
VERSE_COLORS = [
    (255, 180, 50),    # Light blue
    (50, 220, 130),    # Green
    (80, 130, 255),    # Orange-red
    (220, 100, 255),   # Pink
    (50, 255, 255),    # Yellow
    (255, 100, 100),   # Blue
    (100, 255, 200),   # Light green
    (200, 50, 255),    # Magenta
    (255, 200, 100),   # Cyan-ish
    (100, 200, 255),   # Light orange
    (180, 255, 50),    # Lime
    (255, 150, 200),   # Lavender
]

WINDOW_NAME = "Quran Verse Marker"
HELP_TEXT = [
    "Controls:",
    "  Drag   : Draw rect",
    "  N      : Next ayat",
    "  P      : Prev ayat",
    "  U      : Undo rect",
    "  R      : Reset ayat",
    "  S      : Save JSON",
    "  Q/ESC  : Quit",
]


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


class QuranMarker:
    def __init__(self, image_path, surah, start_ayat, output_path=None):
        self.image_path = image_path
        self.surah = surah
        self.start_ayat = start_ayat
        self.current_ayat = start_ayat
        self.current_surah = surah

        # Load image
        self.original = load_image(image_path)
        if self.original is None:
            print(f"Error: Cannot load image '{image_path}'")
            sys.exit(1)

        self.img_h, self.img_w = self.original.shape[:2]
        if self.img_w != 800 or self.img_h != 1293:
            print(f"    ⚠️ Resizing image from {self.img_w}x{self.img_h} to 800x1293 to match standard coordinates system")
            self.original = cv2.resize(self.original, (800, 1293), interpolation=cv2.INTER_AREA)
            self.img_h, self.img_w = self.original.shape[:2]

        # Calculate display scale to fit screen nicely
        screen_h = 900  # reasonable default
        self.scale = min(1.0, screen_h / self.img_h)
        self.disp_w = int(self.img_w * self.scale)
        self.disp_h = int(self.img_h * self.scale)

        # Output path
        if output_path:
            self.output_path = output_path
        else:
            stem = Path(image_path).stem
            self.output_path = str(Path(image_path).parent / f"{stem}_highlights.json")

        # Extract page number from filename if possible
        self.page = self._extract_page_number(image_path)

        # Data: dict of ayat_key -> list of rects
        # ayat_key = "surah:ayat" e.g. "2:17"
        self.verses = {}

        # Drawing state
        self.drawing = False
        self.start_point = None
        self.current_rect = None

        # Load existing data if available
        self._load_existing()

    def _extract_page_number(self, path):
        """Try to extract page number from filename like QK_004.jpg"""
        stem = Path(path).stem
        digits = ''.join(c for c in stem if c.isdigit())
        if digits:
            return int(digits)
        return 0

    def _ayat_key(self, surah=None, ayat=None):
        """Generate key for current or specified verse"""
        s = surah if surah is not None else self.current_surah
        a = ayat if ayat is not None else self.current_ayat
        return f"{s}:{a}"

    def _get_color(self, ayat_key):
        """Get color for a verse based on its index"""
        keys = sorted(self.verses.keys(), key=lambda k: (int(k.split(':')[0]), int(k.split(':')[1])))
        if ayat_key in keys:
            idx = keys.index(ayat_key)
        else:
            idx = len(keys)
        return VERSE_COLORS[idx % len(VERSE_COLORS)]

    def _load_existing(self):
        """Load existing JSON data if file exists"""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as f:
                    data = json.load(f)

                for verse in data.get('verses', []):
                    s = verse.get('surah', self.surah)
                    a = verse.get('ayat', 0)
                    key = f"{s}:{a}"
                    rects = []
                    for h in verse.get('highlights', []):
                        rects.append({
                            'left': h['left'],
                            'top': h['top'],
                            'width': h['width'],
                            'height': h['height']
                        })
                    if rects:
                        self.verses[key] = rects

                print(f"Loaded existing data: {len(self.verses)} verses from {self.output_path}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load existing JSON: {e}")

    def save(self):
        """Save current data to JSON"""
        verses_list = []
        for key in sorted(self.verses.keys(), key=lambda k: (int(k.split(':')[0]), int(k.split(':')[1]))):
            s, a = key.split(':')
            rects = self.verses[key]
            if rects:  # Only save verses with highlights
                verses_list.append({
                    'surah': int(s),
                    'ayat': int(a),
                    'highlights': [
                        {
                            'left': r['left'],
                            'top': r['top'],
                            'width': r['width'],
                            'height': r['height']
                        }
                        for r in rects
                    ]
                })

        data = {
            'page': self.page,
            'image': os.path.basename(self.image_path),
            'image_width': self.img_w,
            'image_height': self.img_h,
            'surah': self.surah,
            'verses': verses_list
        }

        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(verses_list)} verses to {self.output_path}")

    def _to_img_coords(self, x, y):
        """Convert display coordinates to original image coordinates"""
        return int(x / self.scale), int(y / self.scale)

    def _to_disp_coords(self, x, y):
        """Convert original image coordinates to display coordinates"""
        return int(x * self.scale), int(y * self.scale)

    def _draw_overlay(self):
        """Draw all highlights on the image"""
        display = cv2.resize(self.original.copy(), (self.disp_w, self.disp_h))
        overlay = display.copy()

        current_key = self._ayat_key()

        # Draw all existing verse highlights
        for key, rects in self.verses.items():
            color = self._get_color(key)
            is_current = (key == current_key)
            alpha = 0.35 if is_current else 0.2

            for r in rects:
                dx, dy = self._to_disp_coords(r['left'], r['top'])
                dw, dh = int(r['width'] * self.scale), int(r['height'] * self.scale)

                # Fill rectangle
                cv2.rectangle(overlay, (dx, dy), (dx + dw, dy + dh), color, -1)

                # Border
                border_thickness = 2 if is_current else 1
                cv2.rectangle(display, (dx, dy), (dx + dw, dy + dh), color, border_thickness)

            # Blend overlay
            cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
            overlay = display.copy()

            # Draw ayat label on first rect of each verse
            if rects:
                r = rects[0]
                dx, dy = self._to_disp_coords(r['left'], r['top'])
                s, a = key.split(':')
                label = f"{s}:{a}"
                font_scale = 0.5
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # Label background
                cv2.rectangle(display, (dx, dy - th - 6), (dx + tw + 6, dy), color, -1)
                cv2.putText(display, label, (dx + 3, dy - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Draw current drawing rect if any
        if self.current_rect:
            x1, y1, x2, y2 = self.current_rect
            color = self._get_color(current_key)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Draw info panel
        self._draw_info_panel(display)

        return display

    def _draw_info_panel(self, display):
        """Draw info panel with current state and controls"""
        panel_w = 200
        panel_h = 250
        panel_x = 5
        panel_y = 5

        # Semi-transparent background
        sub = display[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w].copy()
        bg = np.zeros_like(sub)
        bg[:] = (40, 40, 40)
        cv2.addWeighted(bg, 0.8, sub, 0.2, 0, sub)
        display[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = sub

        # Current ayat info
        current_key = self._ayat_key()
        color = self._get_color(current_key)
        num_rects = len(self.verses.get(current_key, []))
        total_verses = len(self.verses)

        y = panel_y + 20
        cv2.putText(display, f"Surah: {self.current_surah}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22
        cv2.putText(display, f"Ayat: {self.current_ayat}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 22
        cv2.putText(display, f"Rects: {num_rects}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22
        cv2.putText(display, f"Total verses: {total_verses}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 22
        cv2.putText(display, f"Page: {self.page}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw separator
        y += 10
        cv2.line(display, (panel_x + 10, y), (panel_x + panel_w - 10, y), (100, 100, 100), 1)
        y += 5

        # Help text
        for text in HELP_TEXT:
            y += 16
            cv2.putText(display, text, (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing rectangles"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_rect = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Center the fixed block height on the y drag
            disp_h = int(BLOCK_HEIGHT * self.scale)
            y_center = (self.start_point[1] + y) / 2
            y1 = int(y_center - disp_h / 2)
            y2 = int(y_center + disp_h / 2)
            
            self.current_rect = (
                min(self.start_point[0], x),
                y1,
                max(self.start_point[0], x),
                y2
            )

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.start_point and (abs(x - self.start_point[0]) > 5 or abs(y - self.start_point[1]) > 5):
                # Convert X to image coordinates
                x1, _ = self._to_img_coords(min(self.start_point[0], x), 0)
                x2, _ = self._to_img_coords(max(self.start_point[0], x), 0)
                
                # Center Y on original image coordinates
                y_center_disp = (self.start_point[1] + y) / 2
                _, y_center_img = self._to_img_coords(0, y_center_disp)
                
                y1_img = int(y_center_img - BLOCK_HEIGHT / 2)

                rect = {
                    'left': x1,
                    'top': y1_img,
                    'width': x2 - x1,
                    'height': BLOCK_HEIGHT
                }

                key = self._ayat_key()
                if key not in self.verses:
                    self.verses[key] = []
                self.verses[key].append(rect)

                print(f"  Added rect to {key}: left={x1}, top={y1_img}, w={x2-x1}, h={BLOCK_HEIGHT}")

            self.current_rect = None
            self.start_point = None

    def run(self):
        """Main loop"""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        print(f"\n{'='*50}")
        print(f"  Quran Verse Marker")
        print(f"  Image: {self.image_path}")
        print(f"  Size: {self.img_w}x{self.img_h}")
        print(f"  Starting: Surah {self.surah}, Ayat {self.start_ayat}")
        print(f"  Output: {self.output_path}")
        print(f"{'='*50}\n")

        while True:
            display = self._draw_overlay()
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                self.save()
                print("\nDone! Saved and exiting.")
                break

            elif key == ord('n'):  # Next ayat
                self.current_ayat += 1
                print(f"\n>> Ayat {self.current_surah}:{self.current_ayat}")

            elif key == ord('p'):  # Previous ayat
                if self.current_ayat > 1:
                    self.current_ayat -= 1
                    print(f"\n>> Ayat {self.current_surah}:{self.current_ayat}")

            elif key == ord('u'):  # Undo
                key_name = self._ayat_key()
                if key_name in self.verses and self.verses[key_name]:
                    removed = self.verses[key_name].pop()
                    print(f"  Undo: removed rect from {key_name}")
                    if not self.verses[key_name]:
                        del self.verses[key_name]

            elif key == ord('r'):  # Reset current ayat
                key_name = self._ayat_key()
                if key_name in self.verses:
                    count = len(self.verses[key_name])
                    del self.verses[key_name]
                    print(f"  Reset: removed {count} rects from {key_name}")

            elif key == ord('s'):  # Save
                self.save()

            elif key == ord('+') or key == ord('='):  # Next surah
                self.current_surah += 1
                self.current_ayat = 1
                print(f"\n>> Surah {self.current_surah}, Ayat {self.current_ayat}")

            elif key == ord('-'):  # Previous surah
                if self.current_surah > 1:
                    self.current_surah -= 1
                    self.current_ayat = 1
                    print(f"\n>> Surah {self.current_surah}, Ayat {self.current_ayat}")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Quran Verse Highlight Marker Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('image', help="Path to Quran page image (JPG, PNG, WebP)")
    parser.add_argument('--surah', type=int, default=1, help="Surah number (default: 1)")
    parser.add_argument('--start-ayat', type=int, default=1, help="Starting ayat number (default: 1)")
    parser.add_argument('--output', type=str, default=None, help="Output JSON path (default: <image>_highlights.json)")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)

    marker = QuranMarker(
        image_path=args.image,
        surah=args.surah,
        start_ayat=args.start_ayat,
        output_path=args.output
    )
    marker.run()


if __name__ == '__main__':
    main()
