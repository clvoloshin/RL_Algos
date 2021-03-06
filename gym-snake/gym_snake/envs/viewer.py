from gym.envs.classic_control.rendering import *
from gym.envs.classic_control.rendering import _add_attrs
from pyglet.gl import *

class newViewer(Viewer):
    '''
    Class that extends Viewer implemetation under gym.envs.classic_control.rendering.
    '''
    def __init__(self, width, height, display=None):
        Viewer.__init__(self,width, height, display)

        self.isopen = True

    def draw_point(self, point, **attrs):
        '''
        Allows for one time addition of a point
        '''
        geom = Point_new(point)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

class Point_new(Geom):
    '''
    Define a point v = (x,y) = (x,y,0)
    '''
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self):
        '''
        Render the point
        '''
        glBegin(GL_POINTS)
        glVertex3f(self.v[0], self.v[1], 0.0)
        glEnd()