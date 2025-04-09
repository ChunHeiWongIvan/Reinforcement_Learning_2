import numpy as np

class source:

    # Initialises source location and strength
    def __init__(self, source_x, source_y, source_radioactivity, sd_noise_pct):
        self.source_x = source_x
        self.source_y = source_y
        self.source_radioactivity = source_radioactivity
        self.sd_noise_pct = sd_noise_pct

    def x(self):
        return self.source_x
    
    def y(self):
        return self.source_y
    
    def distance(self, agent_x, agent_y):
        return np.sqrt((agent_x-self.source_x)**2 + (agent_y-self.source_y)**2)
    
    def radiation_level(self, agent_x, agent_y):
        if self.distance(agent_x, agent_y) == 0:
            true_radiation_level = self.source_radioactivity # Limit radioactivity at 0 m distance (not theoretically possible) to radioactivity at 1 m
        else:
            true_radiation_level = self.source_radioactivity / self.distance(agent_x, agent_y)**2
        return true_radiation_level # (Noise is added later in the overall sum)
    
    def radiation_level_plot(self, agent_x, agent_y): # Special function just to plot colour gradient representing radiation levels (array-safe)
        true_radiation_level = self.source_radioactivity / self.distance(agent_x, agent_y)**2
        return true_radiation_level # No need to plot noise, colour gradient meant to represent true radiation levels
    
    
class agent:

    # Initalises agent location and movement distance
    def __init__(self, init_x, init_y, search_area_x, search_area_y, moveDist):
        # Stores agent initial position
        self.init_x = init_x
        self.init_y = init_y

        # Initialises variables for agent
        self.agent_x = init_x
        self.agent_y = init_y
        self.moveDist = moveDist
        self.search_area_x = search_area_x
        self.search_area_y = search_area_y

        # Initalises move counter to 0
        self.moveCount = 0

        # Initialises actionPossible flag to True
        self.actPossible = True

        # Updates initial agent state
        self.update_state()

    def x(self):
        return self.agent_x
    
    def y(self):
        return self.agent_y
    
    def state(self):
        return self.agent_state
    
    def count(self):
        return self.moveCount
    
    def moveUp(self):
        if self.agent_y <= self.search_area_y - self.moveDist:
            self.agent_y += self.moveDist
            self.update_state()
        else:
            self.actPossible = False
        self.moveCount += 1                   

    def moveDown(self):
        if self.agent_y >= 0 + self.moveDist:
            self.agent_y -= self.moveDist
            self.update_state()
        else:
            self.actPossible = False
        self.moveCount += 1

    def moveLeft(self):
        if self.agent_x >= 0 + self.moveDist:
            self.agent_x -= self.moveDist
            self.update_state()
        else:
            self.actPossible = False
        self.moveCount += 1

    def moveRight(self):
        if self.agent_x <= self.search_area_x - self.moveDist:
            self.agent_x += self.moveDist
            self.update_state()
        else:
            self.actPossible = False
        self.moveCount += 1

    def reset(self):
        self.agent_x = self.init_x
        self.agent_y = self.init_y
        self.update_state()
        self.moveCount = 0
        self.actPossible = True
        return self.agent_state

    def update_state(self):
        self.agent_state = self.agent_y * self.search_area_x + self.agent_x + 1

    def actionPossible(self):
        return self.actPossible