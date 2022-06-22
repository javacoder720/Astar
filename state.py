import math

class State:
    def __init__(self, block):
        self.location = block
        self.g = math.inf
        self.f = 0
        self.h = 0
        self.parent = None

    #setters
    def set_g(self, g):
        self.g = g
        self.f = self.g+self.h

    def set_h(self, h):
        self.h = h
        self.f = self.g+self.h

    def set_f(self, g, h):
        self.g = g
        self.h = h
        self.f = g + h

    def set_parent(self, parent):
        self.parent = parent

    #override some methods
    def __lt__(self, other):
        if self.f == other.f:
            return self.g < other.g
        return self.f < other.f

    def __eq__(self, other):
        if other is None:
            return False
        return self.location == other.location and self.f == other.f

    def __str__(self):
        return str(self.location)
