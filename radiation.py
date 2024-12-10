import numpy as np

class source:

    # Initialises source location and strength
    def __init__(self, source_x, source_y, source_radioactivity):
        self.source_x = source_x
        self.source_y = source_y
        self.source_radioactivity = source_radioactivity

    def x(self):
        return self.source_x
    
    def y(self):
        return self.source_y
    
    def distance(self, agent_x, agent_y):
        return np.sqrt((agent_x-self.source_x)**2+(agent_y-self.source_y)**2)
    
    def radiation_level(self, agent_x, agent_y):
        return self.source_radioactivity / self.distance(agent_x, agent_y)**2
    
class agent:

    # Initalises agent location and movement distance
    def __init__(self, agent_x, agent_y, search_area_x, search_area_y, moveDist):
        self.agent_x = agent_x
        self.agent_y = agent_y
        self.moveDist = moveDist
        self.search_area_x = search_area_x
        self.search_area_y = search_area_y

    def x(self):
        return self.agent_x
    
    def y(self):
        return self.agent_y
    
    def moveUp(self):
        if self.agent_y <= self.search_area_y - self.moveDist:
            self.agent_y += self.moveDist

    def moveDown(self):
        if self.agent_y >= 0 + self.moveDist:
            self.agent_y -= self.moveDist

    def moveLeft(self):
        if self.agent_x >= 0 + self.moveDist:
            self.agent_x -= self.moveDist

    def moveRight(self):
        if self.agent_x <= self.search_area_x - self.moveDist:
            self.agent_x += self.moveDist

