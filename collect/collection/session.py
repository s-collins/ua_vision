class Session:

    def __init__(self):
        self.image = None
        self.boxes = []
        self.selected = None
        self.cross = (0, 0)
        self.pt1 = None
        self.pt2 = None

    def reset(self):
        self.image = None
        self.boxes = []
        self.selected = None
        self.cross = (0, 0)
        self.pt1 = None
        self.pt2 = None

    def delete_selected_label(self):
        if self.selected is not None:
            del self.boxes[self.selected]
            if self.selected != 0:
                self.selected -= 1
            else:
                self.selected = None

    def add_box_point(self, point):
        if self.pt1:
            self.pt2 = point
            self.boxes.append((self.pt1, self.pt2))
            self.pt1 = self.pt2 = None
        else:
            self.pt1 = point
