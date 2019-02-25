#!/usr/bin/python
from collection import controller, camera

if __name__ == '__main__':
	cam = camera.Camera()
	cntrl = controller.Controller(cam)
	cntrl.run()
