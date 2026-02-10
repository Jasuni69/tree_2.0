"""
Visual review interface for label audit results.

Shows suspected mislabels side-by-side:
- Left: Current label's tree photos
- Center: The suspect photo
- Right: Predicted label's tree photos

Run: python tools/audit_review.py
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
import pandas as pd


class AuditReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Label Audit Review")
        self.root.geometry("1600x900")

        # Paths
        self.image_base = Path(r'E:\tree_id_2.0\images')
        self.data_dir = Path(r'E:\tree_id_2.0\data')

        # Find latest audit file
        audit_files = sorted(self.data_dir.glob('label_audit_*_mismatches.xlsx'))
        if not audit_files:
            raise FileNotFoundError("No audit mismatch files found")
        self.audit_file = audit_files[-1]

        # Load data
        self.mismatches = pd.read_excel(self.audit_file)
        self.mismatches = self.mismatches.sort_values('sim_diff', ascending=False).reset_index(drop=True)

        # Load full training data for tree photos
        self.training_data = pd.read_excel(self.data_dir / 'training_data_cleaned.xlsx')

        self.current_idx = 0
        self.decisions = {}  # idx -> 'keep' | 'change' | 'skip'

        self.setup_ui()
        self.show_current()

    def setup_ui(self):
        # Top info bar
        self.info_frame = ttk.Frame(self.root)
        self.info_frame.pack(fill='x', padx=10, pady=5)

        self.progress_label = ttk.Label(self.info_frame, text="", font=('Arial', 12))
        self.progress_label.pack(side='left')

        self.stats_label = ttk.Label(self.info_frame, text="", font=('Arial', 10))
        self.stats_label.pack(side='right')

        # Main content - 3 columns
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Left: Current label tree
        self.left_frame = ttk.LabelFrame(self.main_frame, text="Current Label")
        self.left_frame.pack(side='left', fill='both', expand=True, padx=5)

        self.left_header = ttk.Label(self.left_frame, text="", font=('Arial', 11, 'bold'))
        self.left_header.pack(pady=5)

        self.left_sim = ttk.Label(self.left_frame, text="", font=('Arial', 10))
        self.left_sim.pack()

        self.left_canvas = tk.Canvas(self.left_frame, bg='#333')
        self.left_canvas.pack(fill='both', expand=True, pady=5)

        # Center: Suspect photo
        self.center_frame = ttk.LabelFrame(self.main_frame, text="Suspect Photo")
        self.center_frame.pack(side='left', fill='both', expand=True, padx=5)

        self.center_header = ttk.Label(self.center_frame, text="", font=('Arial', 11, 'bold'))
        self.center_header.pack(pady=5)

        self.center_canvas = tk.Canvas(self.center_frame, bg='#222')
        self.center_canvas.pack(fill='both', expand=True, pady=5)

        # Right: Predicted label tree
        self.right_frame = ttk.LabelFrame(self.main_frame, text="Predicted Label")
        self.right_frame.pack(side='left', fill='both', expand=True, padx=5)

        self.right_header = ttk.Label(self.right_frame, text="", font=('Arial', 11, 'bold'))
        self.right_header.pack(pady=5)

        self.right_sim = ttk.Label(self.right_frame, text="", font=('Arial', 10))
        self.right_sim.pack()

        self.right_canvas = tk.Canvas(self.right_frame, bg='#333')
        self.right_canvas.pack(fill='both', expand=True, pady=5)

        # Bottom controls
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(fill='x', padx=10, pady=10)

        # Navigation
        self.nav_frame = ttk.Frame(self.control_frame)
        self.nav_frame.pack(side='left')

        ttk.Button(self.nav_frame, text="<< Prev (A)", command=self.prev_item).pack(side='left', padx=2)
        ttk.Button(self.nav_frame, text="Next (D) >>", command=self.next_item).pack(side='left', padx=2)

        self.jump_var = tk.StringVar()
        ttk.Entry(self.nav_frame, textvariable=self.jump_var, width=6).pack(side='left', padx=5)
        ttk.Button(self.nav_frame, text="Go", command=self.jump_to).pack(side='left')

        # Decision buttons
        self.decision_frame = ttk.Frame(self.control_frame)
        self.decision_frame.pack(side='right')

        ttk.Button(self.decision_frame, text="Keep Current (1)",
                   command=lambda: self.record_decision('keep')).pack(side='left', padx=5)
        ttk.Button(self.decision_frame, text="Change to Predicted (2)",
                   command=lambda: self.record_decision('change')).pack(side='left', padx=5)
        ttk.Button(self.decision_frame, text="Skip/Unsure (3)",
                   command=lambda: self.record_decision('skip')).pack(side='left', padx=5)

        ttk.Button(self.decision_frame, text="Export Decisions",
                   command=self.export_decisions).pack(side='left', padx=20)

        # Key bindings
        self.root.bind('a', lambda e: self.prev_item())
        self.root.bind('d', lambda e: self.next_item())
        self.root.bind('1', lambda e: self.record_decision('keep'))
        self.root.bind('2', lambda e: self.record_decision('change'))
        self.root.bind('3', lambda e: self.record_decision('skip'))
        self.root.bind('<Left>', lambda e: self.prev_item())
        self.root.bind('<Right>', lambda e: self.next_item())

    def get_tree_photos(self, tree_key, exclude_path=None, max_photos=6):
        """Get sample photos for a tree."""
        tree_photos = self.training_data[self.training_data['key'] == tree_key]
        if exclude_path:
            tree_photos = tree_photos[tree_photos['image_path'] != exclude_path]
        return tree_photos['image_path'].head(max_photos).tolist()

    def load_image(self, path, max_size=(300, 300)):
        """Load and resize image."""
        try:
            full_path = self.image_base / path
            img = Image.open(full_path)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def display_photos_grid(self, canvas, photo_paths, max_size=(200, 200)):
        """Display multiple photos in a grid on canvas."""
        canvas.delete('all')
        self._photo_refs = getattr(self, '_photo_refs', [])

        canvas.update()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        if not photo_paths:
            canvas.create_text(cw//2, ch//2, text="No other photos", fill='white', font=('Arial', 12))
            return

        # Calculate grid
        n = len(photo_paths)
        cols = min(2, n)
        rows = (n + cols - 1) // cols

        cell_w = cw // cols
        cell_h = ch // rows
        thumb_size = (min(cell_w - 10, max_size[0]), min(cell_h - 10, max_size[1]))

        for i, path in enumerate(photo_paths):
            row = i // cols
            col = i % cols
            x = col * cell_w + cell_w // 2
            y = row * cell_h + cell_h // 2

            img = self.load_image(path, thumb_size)
            if img:
                self._photo_refs.append(img)
                canvas.create_image(x, y, image=img, anchor='center')

    def show_current(self):
        """Display current mismatch."""
        if self.current_idx >= len(self.mismatches):
            return

        row = self.mismatches.iloc[self.current_idx]

        # Update progress
        self.progress_label.config(
            text=f"Item {self.current_idx + 1} / {len(self.mismatches)}  |  "
                 f"Diff: +{row['sim_diff']:.3f}"
        )

        # Stats
        decided = len(self.decisions)
        keeps = sum(1 for v in self.decisions.values() if v == 'keep')
        changes = sum(1 for v in self.decisions.values() if v == 'change')
        skips = sum(1 for v in self.decisions.values() if v == 'skip')
        self.stats_label.config(
            text=f"Decided: {decided} | Keep: {keeps} | Change: {changes} | Skip: {skips}"
        )

        # Left: Current label
        current_key = row['current_key']
        self.left_header.config(text=current_key)
        self.left_sim.config(text=f"Similarity: {row['current_sim']:.3f}")

        current_photos = self.get_tree_photos(current_key, exclude_path=row['image_path'])
        self.root.after(10, lambda: self.display_photos_grid(self.left_canvas, current_photos))

        # Center: Suspect photo
        self.center_header.config(text=row['image_path'])

        self.center_canvas.delete('all')
        self.center_canvas.update()
        cw = self.center_canvas.winfo_width()
        ch = self.center_canvas.winfo_height()

        suspect_img = self.load_image(row['image_path'], (cw - 20, ch - 20))
        if suspect_img:
            self._suspect_img = suspect_img
            self.center_canvas.create_image(cw//2, ch//2, image=suspect_img, anchor='center')

        # Highlight if decision made
        decision = self.decisions.get(self.current_idx)
        if decision == 'keep':
            self.center_frame.config(text="Suspect Photo [KEEP]")
        elif decision == 'change':
            self.center_frame.config(text="Suspect Photo [CHANGE]")
        elif decision == 'skip':
            self.center_frame.config(text="Suspect Photo [SKIP]")
        else:
            self.center_frame.config(text="Suspect Photo")

        # Right: Predicted label
        pred_key = row['predicted_key']
        self.right_header.config(text=pred_key)
        self.right_sim.config(text=f"Similarity: {row['predicted_sim']:.3f}")

        pred_photos = self.get_tree_photos(pred_key, exclude_path=row['image_path'])
        self.root.after(10, lambda: self.display_photos_grid(self.right_canvas, pred_photos))

    def prev_item(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current()

    def next_item(self):
        if self.current_idx < len(self.mismatches) - 1:
            self.current_idx += 1
            self.show_current()

    def jump_to(self):
        try:
            idx = int(self.jump_var.get()) - 1
            if 0 <= idx < len(self.mismatches):
                self.current_idx = idx
                self.show_current()
        except ValueError:
            pass

    def record_decision(self, decision):
        self.decisions[self.current_idx] = decision
        self.show_current()  # Update display
        self.next_item()  # Auto-advance

    def export_decisions(self):
        if not self.decisions:
            print("No decisions to export")
            return

        # Add decisions to dataframe
        export_df = self.mismatches.copy()
        export_df['decision'] = export_df.index.map(lambda i: self.decisions.get(i, ''))

        # Filter to only decided items
        decided_df = export_df[export_df['decision'] != '']

        output_path = self.data_dir / 'audit_decisions.xlsx'
        decided_df.to_excel(output_path, index=False)
        print(f"Exported {len(decided_df)} decisions to {output_path}")

        # Summary
        print(f"\nSummary:")
        print(f"  Keep current: {sum(1 for v in self.decisions.values() if v == 'keep')}")
        print(f"  Change to predicted: {sum(1 for v in self.decisions.values() if v == 'change')}")
        print(f"  Skipped: {sum(1 for v in self.decisions.values() if v == 'skip')}")


def main():
    root = tk.Tk()
    app = AuditReviewApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
