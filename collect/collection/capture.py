import Tkinter as tk
from PIL import Image, ImageTk
import copy
import cv2 as cv


class CaptureTabPresenter:

    labels = {
        'frame': 'Camera Feed',
        'button': 'Capture Frame'
    }

    def __init__(self):
        self.frame = None
        self.observers = []

    def update_feed(self, feed):
        self.frame = copy.deepcopy(feed)
        self.notify_all()

    def register(self, observer):
        self.observers.append(observer)

    def notify_all(self):
        for observer in self.observers:
            observer.notify()

class CaptureTabView(tk.Frame):

    def __init__(self, master, controller, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.controller = controller
        self.presenter = CaptureTabPresenter()
        self.presenter.register(self)

        self.widgets = {}
        self.widgets['frame'] = tk.LabelFrame(self, text=self.presenter.labels['frame'], padx=10, pady=10)
        self.widgets['feed'] = tk.Label(self.widgets['frame'])
        self.widgets['button'] = tk.Button(self.widgets['frame'], text=self.presenter.labels['button'])

        self.widgets['frame'].grid(padx=10, pady=10)
        self.widgets['feed'].grid(row=0, column=0)
        self.widgets['button'].grid(row=1, column=0, sticky='we')

        self.widgets['button'].config(command=self.controller.capture_frame)

    def notify(self):
        im = cv.cvtColor(self.presenter.frame, cv.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        im = ImageTk.PhotoImage(im)
        self.widgets['feed'].configure(image=im)
        self.widgets['feed'].image = im
