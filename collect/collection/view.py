import Tkinter as tk
import ttk
from capture import *
from edit import *


class View:

    REFRESH_RATE = 200

    def __init__(self, controller):
        self.controller = controller

        self.root = tk.Tk()
        self.notebook = ttk.Notebook(self.root)
        self.capture_tab = CaptureTabView(self.notebook, self.controller)
        self.edit_tab = EditTabView(self.notebook, self.controller)

        self.notebook.add(self.capture_tab, text='Capture')
        self.notebook.add(self.edit_tab, text='Edit')

        self.notebook.grid()

        self.root.after(self.REFRESH_RATE, self.controller.update_feed)

    def activate_capture_tab(self):
        self.notebook.select(0)

    def activate_edit_tab(self):
        self.notebook.select(1)

    def show(self):
        self.root.mainloop()
