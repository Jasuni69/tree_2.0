"""
Tkinter live progress window for training monitoring.
Runs in a separate thread so it doesn't block training.

Usage:
    from progress_window import TrainingProgressWindow
    win = TrainingProgressWindow(total_epochs=60)
    win.update_epoch(epoch=1, arc_loss=5.2, tri_loss=0.8, lr=1e-3)
    win.update_batch(batch=10, total_batches=100)
    win.update_val(recall_1=0.45, recall_5=0.72)
    win.close()
"""

import threading
import tkinter as tk
from tkinter import ttk
import time
from collections import deque


class TrainingProgressWindow:
    def __init__(self, total_epochs: int = 60, title: str = "Tree Re-ID Training"):
        self.total_epochs = total_epochs
        self.title = title
        self._running = True
        self._ready = threading.Event()

        # Data for UI updates (thread-safe via queue)
        self._updates = deque(maxlen=100)
        self._loss_history = []
        self._recall_history = []

        # State
        self._current_epoch = 0
        self._current_batch = 0
        self._total_batches = 0
        self._arc_loss = 0.0
        self._tri_loss = 0.0
        self._lr = 0.0
        self._best_recall = 0.0
        self._phase = "Phase 1 (frozen)"
        self._recall_1 = 0.0
        self._recall_5 = 0.0
        self._start_time = time.time()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self):
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("520x480")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Title.TLabel", font=("Consolas", 14, "bold"),
                         foreground="#cdd6f4", background="#1e1e2e")
        style.configure("Info.TLabel", font=("Consolas", 11),
                         foreground="#a6adc8", background="#1e1e2e")
        style.configure("Value.TLabel", font=("Consolas", 12, "bold"),
                         foreground="#89b4fa", background="#1e1e2e")
        style.configure("Good.TLabel", font=("Consolas", 12, "bold"),
                         foreground="#a6e3a1", background="#1e1e2e")
        style.configure("Phase.TLabel", font=("Consolas", 10),
                         foreground="#f9e2af", background="#1e1e2e")
        style.configure("Custom.Horizontal.TProgressbar",
                         troughcolor="#313244", background="#89b4fa")

        main = tk.Frame(self.root, bg="#1e1e2e", padx=20, pady=15)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main, text="Tree Re-ID Metric Learning",
                  style="Title.TLabel").pack(anchor="w")

        # Phase
        self.phase_label = ttk.Label(main, text="Phase 1 (frozen backbone)",
                                     style="Phase.TLabel")
        self.phase_label.pack(anchor="w", pady=(2, 10))

        # Epoch progress
        epoch_frame = tk.Frame(main, bg="#1e1e2e")
        epoch_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(epoch_frame, text="Epoch:", style="Info.TLabel").pack(side=tk.LEFT)
        self.epoch_label = ttk.Label(epoch_frame, text="0 / 0", style="Value.TLabel")
        self.epoch_label.pack(side=tk.RIGHT)

        self.epoch_bar = ttk.Progressbar(main, length=480, mode='determinate',
                                          style="Custom.Horizontal.TProgressbar")
        self.epoch_bar.pack(fill=tk.X, pady=(0, 10))

        # Batch progress
        batch_frame = tk.Frame(main, bg="#1e1e2e")
        batch_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(batch_frame, text="Batch:", style="Info.TLabel").pack(side=tk.LEFT)
        self.batch_label = ttk.Label(batch_frame, text="0 / 0", style="Value.TLabel")
        self.batch_label.pack(side=tk.RIGHT)

        self.batch_bar = ttk.Progressbar(main, length=480, mode='determinate',
                                          style="Custom.Horizontal.TProgressbar")
        self.batch_bar.pack(fill=tk.X, pady=(0, 15))

        # Separator
        tk.Frame(main, bg="#45475a", height=1).pack(fill=tk.X, pady=5)

        # Loss section
        loss_frame = tk.Frame(main, bg="#1e1e2e")
        loss_frame.pack(fill=tk.X, pady=(10, 5))
        ttk.Label(loss_frame, text="ArcFace Loss:", style="Info.TLabel").pack(side=tk.LEFT)
        self.arc_label = ttk.Label(loss_frame, text="--", style="Value.TLabel")
        self.arc_label.pack(side=tk.RIGHT)

        tri_frame = tk.Frame(main, bg="#1e1e2e")
        tri_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(tri_frame, text="Triplet Loss:", style="Info.TLabel").pack(side=tk.LEFT)
        self.tri_label = ttk.Label(tri_frame, text="--", style="Value.TLabel")
        self.tri_label.pack(side=tk.RIGHT)

        lr_frame = tk.Frame(main, bg="#1e1e2e")
        lr_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(lr_frame, text="Learning Rate:", style="Info.TLabel").pack(side=tk.LEFT)
        self.lr_label = ttk.Label(lr_frame, text="--", style="Value.TLabel")
        self.lr_label.pack(side=tk.RIGHT)

        # Separator
        tk.Frame(main, bg="#45475a", height=1).pack(fill=tk.X, pady=5)

        # Validation section
        val_frame = tk.Frame(main, bg="#1e1e2e")
        val_frame.pack(fill=tk.X, pady=(10, 5))
        ttk.Label(val_frame, text="Val Recall@1:", style="Info.TLabel").pack(side=tk.LEFT)
        self.r1_label = ttk.Label(val_frame, text="--", style="Good.TLabel")
        self.r1_label.pack(side=tk.RIGHT)

        val5_frame = tk.Frame(main, bg="#1e1e2e")
        val5_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(val5_frame, text="Val Recall@5:", style="Info.TLabel").pack(side=tk.LEFT)
        self.r5_label = ttk.Label(val5_frame, text="--", style="Good.TLabel")
        self.r5_label.pack(side=tk.RIGHT)

        best_frame = tk.Frame(main, bg="#1e1e2e")
        best_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(best_frame, text="Best Recall@1:", style="Info.TLabel").pack(side=tk.LEFT)
        self.best_label = ttk.Label(best_frame, text="--", style="Good.TLabel")
        self.best_label.pack(side=tk.RIGHT)

        # Elapsed time
        time_frame = tk.Frame(main, bg="#1e1e2e")
        time_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(time_frame, text="Elapsed:", style="Info.TLabel").pack(side=tk.LEFT)
        self.time_label = ttk.Label(time_frame, text="00:00:00", style="Info.TLabel")
        self.time_label.pack(side=tk.RIGHT)

        self._ready.set()
        self._poll()
        self.root.mainloop()

    def _poll(self):
        if not self._running:
            self.root.destroy()
            return

        # Process updates
        while self._updates:
            update = self._updates.popleft()
            self._apply_update(update)

        # Update elapsed time
        elapsed = int(time.time() - self._start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        self.time_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")

        self.root.after(200, self._poll)

    def _apply_update(self, update):
        kind = update.get('kind')

        if kind == 'epoch':
            ep = update['epoch']
            self._current_epoch = ep
            self.epoch_label.configure(text=f"{ep} / {self.total_epochs}")
            self.epoch_bar['value'] = (ep / self.total_epochs) * 100
            self.arc_label.configure(text=f"{update.get('arc_loss', 0):.4f}")
            self.tri_label.configure(text=f"{update.get('tri_loss', 0):.4f}")
            self.lr_label.configure(text=f"{update.get('lr', 0):.1e}")

            if ep <= 3:
                self.phase_label.configure(text="Phase 1 (frozen backbone)")
            else:
                self.phase_label.configure(text="Phase 2 (full fine-tune)")

        elif kind == 'batch':
            b = update['batch']
            total = update['total']
            self.batch_label.configure(text=f"{b} / {total}")
            self.batch_bar['value'] = (b / max(total, 1)) * 100
            if 'arc_loss' in update:
                self.arc_label.configure(text=f"{update['arc_loss']:.4f}")
            if 'tri_loss' in update:
                self.tri_label.configure(text=f"{update['tri_loss']:.4f}")

        elif kind == 'val':
            r1 = update.get('recall_1', 0)
            r5 = update.get('recall_5', 0)
            self.r1_label.configure(text=f"{r1:.2%}")
            self.r5_label.configure(text=f"{r5:.2%}")
            if r1 > self._best_recall:
                self._best_recall = r1
            self.best_label.configure(text=f"{self._best_recall:.2%}")

        elif kind == 'status':
            self.phase_label.configure(text=update.get('text', ''))

    def update_epoch(self, epoch: int, arc_loss: float = 0, tri_loss: float = 0, lr: float = 0):
        self._updates.append({
            'kind': 'epoch', 'epoch': epoch,
            'arc_loss': arc_loss, 'tri_loss': tri_loss, 'lr': lr
        })

    def update_batch(self, batch: int, total_batches: int,
                     arc_loss: float = 0, tri_loss: float = 0):
        self._updates.append({
            'kind': 'batch', 'batch': batch, 'total': total_batches,
            'arc_loss': arc_loss, 'tri_loss': tri_loss
        })

    def update_val(self, recall_1: float, recall_5: float):
        self._updates.append({
            'kind': 'val', 'recall_1': recall_1, 'recall_5': recall_5
        })

    def update_status(self, text: str):
        self._updates.append({'kind': 'status', 'text': text})

    def close(self):
        self._running = False
