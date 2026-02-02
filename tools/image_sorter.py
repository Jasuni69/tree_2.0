"""
Pygame-ce GUI tool for sorting training dataset images.
Remove non-tree images fast.
"""

import pygame
import pandas as pd
from pathlib import Path
import json
from threading import Thread
from queue import Queue
from typing import Dict, List, Tuple
import sys

# Config
EXCEL_PATH = Path(r'E:\tree_id_2.0\data\training_data_cleaned.xlsx')
IMAGE_BASE = Path(r'E:\tree_id_2.0\images')
DECISIONS_PATH = Path(r'E:\tree_id_2.0\data\sorting_decisions.json')
WINDOW_SIZE = (1400, 900)
PRELOAD_COUNT = 3

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 50)

# Flash duration (frames)
FLASH_DURATION = 10


class ImagePreloader:
    """Preload images in background for smooth browsing."""

    def __init__(self, image_paths: List[str], base_path: Path):
        self.image_paths = image_paths
        self.base_path = base_path
        self.cache: Dict[int, pygame.Surface] = {}
        self._raw_cache: Dict[int, bytes] = {}
        self.current_index = 0

    def request_preload(self, start_idx: int):
        """Preload images around current index (main thread safe)."""
        self.current_index = start_idx
        # Clear old cache
        to_remove = [k for k in list(self.cache.keys()) if k < start_idx - 5 or k > start_idx + PRELOAD_COUNT + 5]
        for k in to_remove:
            self.cache.pop(k, None)

    def get_image(self, idx: int):
        """Load image on demand (main thread)."""
        if idx in self.cache:
            return self.cache[idx]

        if 0 <= idx < len(self.image_paths):
            img_path = self.base_path / self.image_paths[idx]
            if img_path.exists():
                try:
                    surface = pygame.image.load(str(img_path)).convert()
                    self.cache[idx] = surface
                    return surface
                except Exception as e:
                    print(f"Load error {img_path}: {e}")
        return None

    def stop(self):
        pass


class ImageSorter:
    """Main sorter application."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Tree Image Sorter")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        # Load data
        print("Loading Excel data...")
        self.df = pd.read_excel(EXCEL_PATH)
        self.image_paths = self.df['image_path'].tolist()
        self.total_images = len(self.image_paths)
        print(f"Loaded {self.total_images} images")

        # Load previous decisions
        self.decisions = self._load_decisions()

        # Find first unreviewed image
        self.current_idx = 0
        for i, path in enumerate(self.image_paths):
            if path not in self.decisions:
                self.current_idx = i
                break

        print(f"Starting at image {self.current_idx + 1}/{self.total_images}")

        # Stats
        self.kept_count = sum(1 for v in self.decisions.values() if v == "keep")
        self.removed_count = sum(1 for v in self.decisions.values() if v == "remove")

        # UI state
        self.flash_color = None
        self.flash_frames = 0
        self.zoom_mode = False  # False = fit, True = actual size
        self.zoom_offset = [0, 0]  # Pan offset for zoomed view
        self.dragging = False
        self.drag_start = None

        # Image display area (leave space for info at bottom)
        self.display_area = pygame.Rect(10, 50, WINDOW_SIZE[0] - 20, WINDOW_SIZE[1] - 250)

        # Start preloader
        self.preloader = ImagePreloader(self.image_paths, IMAGE_BASE)
        self.preloader.request_preload(self.current_idx)

        self.running = True

    def _load_decisions(self) -> Dict[str, str]:
        """Load previous sorting decisions."""
        if DECISIONS_PATH.exists():
            with open(DECISIONS_PATH, 'r') as f:
                return json.load(f)
        return {}

    def _save_decisions(self):
        """Save sorting decisions to JSON."""
        DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DECISIONS_PATH, 'w') as f:
            json.dump(self.decisions, f, indent=2)
        print(f"Saved decisions: {len(self.decisions)} images reviewed")

    def _mark_decision(self, decision: str):
        """Mark current image with decision and move to next."""
        current_path = self.image_paths[self.current_idx]
        self.decisions[current_path] = decision

        if decision == "keep":
            self.kept_count += 1
            self.flash_color = GREEN
        else:
            self.removed_count += 1
            self.flash_color = RED

        self.flash_frames = FLASH_DURATION
        self._next_image()

    def _next_image(self):
        """Move to next image."""
        if self.current_idx < self.total_images - 1:
            self.current_idx += 1
            self.preloader.request_preload(self.current_idx)
            self.zoom_offset = [0, 0]

    def _prev_image(self):
        """Move to previous image (undo)."""
        if self.current_idx > 0:
            # Remove decision for current image if exists
            current_path = self.image_paths[self.current_idx]
            if current_path in self.decisions:
                if self.decisions[current_path] == "keep":
                    self.kept_count -= 1
                else:
                    self.removed_count -= 1
                del self.decisions[current_path]

            self.current_idx -= 1
            self.preloader.request_preload(self.current_idx)
            self.zoom_offset = [0, 0]

    def _get_current_image_info(self) -> Dict:
        """Get info for current image."""
        row = self.df.iloc[self.current_idx]
        return {
            'key': row.get('key', 'N/A'),
            'address': row.get('address', 'N/A'),
            'tree_number': row.get('tree_number', 'N/A'),
            'image_path': row['image_path'],
            'cleaning_note': row.get('cleaning_note', '')
        }

    def _scale_image(self, surface: pygame.Surface) -> Tuple[pygame.Surface, pygame.Rect]:
        """Scale image to fit display area while maintaining aspect ratio."""
        if self.zoom_mode:
            # Actual size - no scaling
            rect = surface.get_rect()
            rect.center = (
                self.display_area.centerx + self.zoom_offset[0],
                self.display_area.centery + self.zoom_offset[1]
            )
            return surface, rect
        else:
            # Fit to display area
            img_rect = surface.get_rect()
            scale = min(
                self.display_area.width / img_rect.width,
                self.display_area.height / img_rect.height
            )

            new_size = (int(img_rect.width * scale), int(img_rect.height * scale))
            scaled = pygame.transform.smoothscale(surface, new_size)

            rect = scaled.get_rect()
            rect.center = self.display_area.center
            return scaled, rect

    def _draw_text(self, text: str, pos: Tuple[int, int], font: pygame.font.Font, color=WHITE):
        """Draw text on screen."""
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)

    def _draw_ui(self):
        """Draw all UI elements."""
        self.screen.fill(DARK_GRAY)

        # Draw flash border if active
        if self.flash_frames > 0:
            pygame.draw.rect(self.screen, self.flash_color, (0, 0, WINDOW_SIZE[0], WINDOW_SIZE[1]), 10)
            self.flash_frames -= 1

        # Top bar - stats and progress
        progress_text = f"Image {self.current_idx + 1}/{self.total_images}"
        stats_text = f"Kept: {self.kept_count} | Removed: {self.removed_count}"

        self._draw_text(progress_text, (10, 10), self.font_large)
        self._draw_text(stats_text, (400, 15), self.font_medium, GREEN if self.kept_count > self.removed_count else RED)

        # Load and display current image
        surface = self.preloader.get_image(self.current_idx)

        if surface:
            scaled, rect = self._scale_image(surface)

            # Clip to display area
            self.screen.set_clip(self.display_area)
            self.screen.blit(scaled, rect)
            self.screen.set_clip(None)

            # Draw border around display area
            pygame.draw.rect(self.screen, GRAY, self.display_area, 2)
        else:
            # Image not loaded
            self._draw_text("Loading...", (WINDOW_SIZE[0]//2 - 50, WINDOW_SIZE[1]//2), self.font_large, GRAY)

        # Image info at bottom
        info = self._get_current_image_info()
        y = WINDOW_SIZE[1] - 190

        self._draw_text(f"Key: {info['key']}", (10, y), self.font_medium)
        y += 30
        self._draw_text(f"Address: {info['address']}", (10, y), self.font_small)
        y += 25
        self._draw_text(f"Tree: {info['tree_number']}", (10, y), self.font_small)
        y += 25
        self._draw_text(f"Path: {info['image_path']}", (10, y), self.font_small, GRAY)
        y += 25

        if info['cleaning_note']:
            self._draw_text(f"Note: {info['cleaning_note']}", (10, y), self.font_small, (255, 255, 0))

        # Controls help at bottom
        help_y = WINDOW_SIZE[1] - 40
        help_text = "RIGHT/D: Keep | LEFT/A: Remove | UP/W: Undo | S: Save | Q/ESC: Quit | SPACE: Zoom"
        self._draw_text(help_text, (10, help_y), self.font_small, GRAY)

        # Zoom indicator
        if self.zoom_mode:
            zoom_text = "ZOOM: Actual Size (Drag to pan)"
            self._draw_text(zoom_text, (WINDOW_SIZE[0] - 350, 15), self.font_medium, (255, 255, 0))

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                # Keep image
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    self._mark_decision("keep")

                # Remove image
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    self._mark_decision("remove")

                # Undo (go back)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    self._prev_image()

                # Save
                elif event.key == pygame.K_s:
                    self._save_decisions()

                # Quit
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False

                # Toggle zoom
                elif event.key == pygame.K_SPACE:
                    self.zoom_mode = not self.zoom_mode
                    self.zoom_offset = [0, 0]

            # Mouse drag for panning in zoom mode
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.zoom_mode and event.button == 1:  # Left click
                    self.dragging = True
                    self.drag_start = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging and self.zoom_mode:
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]
                    self.zoom_offset[0] += dx
                    self.zoom_offset[1] += dy
                    self.drag_start = event.pos

    def run(self):
        """Main application loop."""
        print("\nControls:")
        print("  RIGHT or D: Keep image")
        print("  LEFT or A: Remove image")
        print("  UP or W: Go back (undo)")
        print("  S: Save progress")
        print("  Q or ESC: Save and quit")
        print("  SPACE: Toggle zoom")
        print("\nStarting...")

        while self.running:
            self._handle_events()
            self._draw_ui()
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

        # Cleanup
        print("\nShutting down...")
        self._save_decisions()
        self.preloader.stop()

        # Print summary
        reviewed = len(self.decisions)
        print(f"\nSummary:")
        print(f"  Total reviewed: {reviewed}/{self.total_images}")
        print(f"  Kept: {self.kept_count}")
        print(f"  Removed: {self.removed_count}")
        print(f"  Saved to: {DECISIONS_PATH}")

        pygame.quit()


def main():
    """Entry point."""
    # Check files exist
    if not EXCEL_PATH.exists():
        print(f"ERROR: Excel file not found at {EXCEL_PATH}")
        sys.exit(1)

    if not IMAGE_BASE.exists():
        print(f"ERROR: Image base directory not found at {IMAGE_BASE}")
        sys.exit(1)

    sorter = ImageSorter()
    sorter.run()


if __name__ == '__main__':
    main()
