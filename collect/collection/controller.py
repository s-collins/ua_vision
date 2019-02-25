from view import View
from session import *
from camera import *
from persistence import *
from training_example import *


class Controller:

    def __init__(self, camera):
        self.camera = camera
        self.persistence = Persistence()
        self.view = View(self)
        self.session = Session()
        self.feed = None

    def run(self):
        self.view.show()

    # CALLBACKS

    def capture_frame(self):
        self.session.image = self.feed
        self.view.edit_tab.presenter.update(self.session)
        self.view.activate_edit_tab()

    def delete_selected_label(self):
        self.session.delete_selected_label()
        self.view.edit_tab.presenter.update(self.session)

    def listbox_selection(self, event):
        selection = event.widget.curselection()
        if selection:
            self.session.selected = selection[0]
        self.view.edit_tab.presenter.update(self.session)

    def mouse_click(self, event):
        self.session.add_box_point((event.x, event.y))
        self.view.edit_tab.presenter.update(self.session)

    def mouse_motion(self, event):
        self.session.cross = (event.x, event.y)
        self.view.edit_tab.presenter.update(self.session)

    def save_button(self):
        properties = self.view.edit_tab.get_properties()
        if properties['populated'] and self.session.image is not None:
            t = TrainingExample()
            t.camera_angle = properties['camera_angle']
            t.camera_height = properties['camera_height']
            t.light_angle = properties['light_angle']
            t.image = self.session.image
            t.labels = self.session.boxes
            self.persistence.save(t)
            self.session.reset()
            self.view.edit_tab.presenter.update(self.session)
            print('Success: Saved training example.')
            self.view.activate_capture_tab()
        else:
            print('Error: Failed to save training example.')

    def update_feed(self):
        self.feed = self.camera.get_frame()
        self.view.capture_tab.presenter.update_feed(self.feed)
        self.view.root.after(self.view.REFRESH_RATE, self.update_feed)

