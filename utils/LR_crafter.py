import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class LRSchedulerUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('LR Scheduler Designer')
        self.geometry('800x600')

        # Plot setup
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Data
        self.lr_points = []  # List of (%training, lr) tuples
        self.selected_point = None
        self.dragging = False

        # Plot initial empty chart
        self.refresh_plot()

        # Mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Export button
        self.export_button = tk.Button(self, text="Export LR Schedule", command=self.export_lr_schedule)
        self.export_button.pack(side=tk.BOTTOM, fill=tk.X)

    def refresh_plot(self):
        self.plot.clear()
        # Plot current LR schedule
        if self.lr_points:
            epochs, lrs = zip(*sorted(self.lr_points, key=lambda x: x[0]))
            self.plot.plot(epochs, lrs, marker='o', linestyle='-')
        self.plot.set_title('Learning Rate Schedule')
        self.plot.set_xlabel('Training')
        self.plot.set_ylabel('Learning Rate')
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.plot.axes: return

        if event.button == 3:
            if self.lr_points:
                len_to_nearest_point = [((x - event.xdata)**2 + (y - event.ydata)**2)**0.5 for x, y in self.lr_points]
                nearest_point = np.argmin(len_to_nearest_point)
                if len_to_nearest_point[nearest_point] < 0.05:
                    self.lr_points.pop(nearest_point)
                    self.refresh_plot()
                self.refresh_plot()
        elif event.button == 1:
            self.dragging = True
            self.lr_points.append((event.xdata, event.ydata))
            self.refresh_plot()
        
    def on_motion(self, event):
        if not self.dragging or self.selected_point is None: return
        if event.inaxes != self.plot.axes: return
        # Update point position
        self.lr_points[self.selected_point] = (event.xdata, event.ydata)
        self.refresh_plot()

    def on_release(self, event):
        self.dragging = False
        self.selected_point = None

    def export_lr_schedule(self):
        # Export the LR schedule in a format your script can use
        sorted_schedule = sorted(self.lr_points, key=lambda x: x[0])
        print(f"Exported LR Schedule: {sorted_schedule}")
        messagebox.showinfo("Export Success", f"LR Schedule exported to console")

if __name__ == "__main__":
    app = LRSchedulerUI()
    app.mainloop()
