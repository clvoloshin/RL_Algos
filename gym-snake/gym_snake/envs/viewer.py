from gym.envs.classic_control.rendering import *
from gym.envs.classic_control.rendering import _add_attrs
from pyglet.gl import *

class newViewer(Viewer):
    def __init__(self, width, height, display=None):
        Viewer.__init__(self,width, height, display)

        self.isopen = True

    def draw_point(self, point, **attrs):
        geom = Point_new(point)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

class Point_new(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        glBegin(GL_POINTS) # draw point
        glVertex3f(self.v[0], self.v[1], 0.0)
        glEnd()