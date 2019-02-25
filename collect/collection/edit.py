import Tkinter as tk
from PIL import Image, ImageTk
import copy
import cv2 as cv


class EditTabPresenter:

    labels = {
        'image_frame': 'Image',
        'label_frame': 'Labels',
        'properties_frame': 'Properties',
        'delete_button': 'Remove Selected',
        'save_button': 'Save Training Example',
        'camera_angle': 'Camera Angle (degrees):',
        'camera_height': '\nCamera Height (cm):',
        'light_angle': '\nLight Angle (degrees):'
    }

    CROSS_COLOR = (0, 0, 255)
    BOX_COLOR = (0, 255, 0)
    SELECTED_BOX_COLOR = (255, 0, 0)

    def __init__(self):
        self.observers = []
        self.image = None
        self.boxes = []
        self.cross_col = 0
        self.cross_row = 0
        self.box_origin = None
        self.selected = None

    # ------------------------------------------------------------------------------------------------------------------
    # Observer Pattern
    # ------------------------------------------------------------------------------------------------------------------

    def register(self, observer):
        self.observers.append(observer)

    def notify_all(self):
        for observer in self.observers:
            observer.notify()

    # ------------------------------------------------------------------------------------------------------------------
    # Mutator Methods
    # ------------------------------------------------------------------------------------------------------------------

    def update(self, session):
        self.image = copy.deepcopy(session.image)
        self.boxes = session.boxes
        self.selected = session.selected
        self.cross_col = session.cross[0]
        self.cross_row = session.cross[1]
        self.box_origin = session.pt1
        self.notify_all()


class EditTabView(tk.Frame):

    def __init__(self, master, controller, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.presenter = EditTabPresenter()
        self.presenter.register(self)
        self.controller = controller

        #  Create the Widgets
        self.widgets = {}
        self.widgets['image_frame'] = tk.LabelFrame(self, text=self.presenter.labels['image_frame'], padx=10, pady=10)
        self.widgets['label_frame'] = tk.LabelFrame(self, text=self.presenter.labels['label_frame'], padx=10, pady=10)
        self.widgets['properties_frame'] = tk.LabelFrame(self, text=self.presenter.labels['properties_frame'], padx=10, pady=10)
        self.widgets['image_lbl'] = tk.Label(self.widgets['image_frame'])
        self.widgets['listbox'] = tk.Listbox(self.widgets['label_frame'], selectbackground='Blue', selectforeground='White')
        self.widgets['delete_button'] = tk.Button(self.widgets['label_frame'], text=self.presenter.labels['delete_button'])

        self.widgets['camera_angle_lbl'] = tk.Label(self.widgets['properties_frame'], text=self.presenter.labels['camera_angle'])
        self.widgets['camera_angle'] = tk.Entry(self.widgets['properties_frame'])

        self.widgets['camera_height_lbl'] = tk.Label(self.widgets['properties_frame'], text=self.presenter.labels['camera_height'])
        self.widgets['camera_height'] = tk.Entry(self.widgets['properties_frame'])

        self.widgets['light_angle_lbl'] = tk.Label(self.widgets['properties_frame'], text=self.presenter.labels['light_angle'])
        self.widgets['light_angle'] = tk.Entry(self.widgets['properties_frame'])


        self.widgets['save_button'] = tk.Button(self, text=self.presenter.labels['save_button'])

        # Layout the Widgets
        self.widgets['image_frame'].grid(row=0, column=0, padx=10, pady=10, sticky='we')
        self.widgets['label_frame'].grid(row=1, column=0, padx=10, pady=10, sticky='we')
        self.widgets['properties_frame'].grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky='nswe')
        self.widgets['image_lbl'].grid(row=0, column=0)
        self.widgets['listbox'].pack(fill=tk.BOTH, expand=True)
        self.widgets['delete_button'].pack(fill=tk.BOTH, expand=True)

        self.widgets['camera_angle_lbl'].pack(anchor='w')
        self.widgets['camera_angle'].pack(fill=tk.BOTH)

        self.widgets['camera_height_lbl'].pack(anchor='w')
        self.widgets['camera_height'].pack(fill=tk.BOTH)

        self.widgets['light_angle_lbl'].pack(anchor='w')
        self.widgets['light_angle'].pack(fill=tk.BOTH)

        self.widgets['save_button'].grid(row=2, column=0, columnspan=2, sticky='we', padx=10, pady=10)

        self.__setup_callbacks()

    def __setup_callbacks(self):
        # delete selected label button
        widget = self.widgets['delete_button']
        callback = self.controller.delete_selected_label
        widget.config(command=callback)

        # mouse motion over image
        widget = self.widgets['image_lbl']
        callback = self.controller.mouse_motion
        widget.bind('<Motion>', callback)

        # mouse press on image
        widget = self.widgets['image_lbl']
        callback = self.controller.mouse_click
        widget.bind('<Button-1>', callback)

        # listbox selection
        widget = self.widgets['listbox']
        callback = self.controller.listbox_selection
        widget.bind('<<ListboxSelect>>', callback)

        # save button
        widget = self.widgets['save_button']
        callback = self.controller.save_button
        widget.config(command=callback)

    def get_properties(self):
        try:
            properties = {
                'populated': True,
                'camera_angle': int(self.widgets['camera_angle'].get()),
                'camera_height': int(self.widgets['camera_height'].get()),
                'light_angle': int(self.widgets['light_angle'].get())
            }
            return properties
        except:
            properties = {
                'populated': False
            }
            return properties

    # ------------------------------------------------------------------------------------------------------------------
    # Observer Pattern
    # ------------------------------------------------------------------------------------------------------------------

    def notify(self):
        p = self.presenter

        # get image from presenter
        im = p.image

        # add cross annotation
        if im is not None:
            height, width, channesl = im.shape
            x = p.cross_col
            y = p.cross_row
            line1 = ((0, y), (width, y))
            line2 = ((x, 0), (x, height))
            cv.line(im, line1[0], line1[1], p.CROSS_COLOR)
            cv.line(im, line2[0], line2[1], p.CROSS_COLOR)

            # add in-progress box
            if p.box_origin:
                cv.rectangle(im, p.box_origin, (p.cross_col, p.cross_row), p.BOX_COLOR)

            # add labels
            for index, box in enumerate(p.boxes):
                color = p.BOX_COLOR
                if index == p.selected:
                    color = p.SELECTED_BOX_COLOR
                cv.rectangle(im, box[0], box[1], color, 2)

            # render image and annotations
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            im = Image.fromarray(im)
            im = ImageTk.PhotoImage(im)
            self.widgets['image_lbl'].configure(image=im)
            self.widgets['image_lbl'].image = im
        else:
            self.widgets['image_lbl'].configure(image=None)
            self.widgets['image_lbl'].image = None

        # update the listbox
        listbox = self.widgets['listbox']
        listbox.delete(0, tk.END)
        for box in p.boxes:
            listbox.insert(tk.END, str(box[0]) + ', ' + str(box[1]))
        if p.selected is not None:
            listbox.selection_set(p.selected)
            listbox.activate(p.selected)
