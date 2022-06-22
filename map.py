import numpy as np
import matplotlib as mpl
import random
import matplotlib.pyplot as plt
from enum import Enum
from math import sqrt

class Map:
    bounds = [-10, 0, 10, 20, 30, 40, 50, 60]

    empty_goal = -5.0
    empty = 5
    hard = 15.0
    hard_river = 25.0
    river = 35.0
    blocked = 45.0
    traversed = 55.0

    def __init__(self, size):
        self.min_row = 0
        self.max_row = size[0]-1
        self.min_col = 0
        self.max_col = size[1]-1

        self.start = None
        self.goal = None
        self.hard_centers = []

        self.grid = np.zeros(size)
        self.grid_copy = np.copy(self.grid)
        self.generate_map()

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def generate_map(self):
        self.generate_partially_blocked()

        for i in range(4):
            self.generate_river()

        self.generate_blocked()

        self.grid_copy = np.copy(self.grid)
        self.generate_start_and_end()

    def generate_partially_blocked(self):
        for i in range(8):
            rand_block = self.random_block()
            self.set_block(rand_block, self.hard)
            self.hard_centers.append(rand_block)
            for neighbor in self.get_neighbors(rand_block, radius=15):
                if random.randint(0, 1) == 0:
                    self.set_block(neighbor, self.hard)

    def generate_river(self):
        self.grid_copy = np.copy(self.grid) #save before generating each river

        dist = 0
        r,c,direction = self.generate_river_start()
        r,c,dist = self.move_in_direction(r,c, direction, 20)

        dist = 0
        while True:
            direction = self.change_direction(direction)
            r,c,d = self.move_in_direction(r,c, direction, 20)
            #print(r,c,d,dist)
            if d < 20:
                if not self.in_range((r,c)) and dist > 100: #hit boundary
                    break
                else:
                    self.reset_map() #resets to what we saved in the beginning
                    r,c,direction = self.generate_river_start()
                    r,c,dist = self.move_in_direction(r,c, direction, 20)
            else:
                dist += d

    def generate_blocked(self):
        count = 0
        while count != int ( .2 * (self.max_col+1) * (self.max_row+1) ):
            r,c = self.random_block(max_threshold = 10)
            self.set_block((r,c), self.blocked)
            count += 1

    def generate_start_and_end(self):
        self.start = self.get_random_block_near_boundary()
        self.goal = self.get_random_block_near_boundary()

        while self.manhattan_distance(self.start, self.goal) < 100:
            self.start = self.get_random_block_near_boundary()
            self.goal = self.get_random_block_near_boundary()

        self.set_start_and_goal(self.start, self.goal)

    def set_start_and_goal(self, start, goal):
        self.start = start
        self.goal = goal
        self.set_block(start, self.traversed)
        self.set_block(goal, self.empty_goal)

    def get_random_block_near_boundary(self):
        random_int = random.randint(1,4)
        r_min, r_max, c_min, c_max = None, None, None, None
        boundary = 20

        if random_int == 1:
            r_min, r_max = 0, boundary
        elif random_int == 2:
            r_min, r_max = self.max_row-boundary, self.max_row
        elif random_int == 3:
            c_min, c_max = 0, boundary
        else:
            c_min, c_max = self.max_col-boundary, self.max_col

        return self.random_block(r_min, r_max, c_min, c_max, max_threshold=20)

    #river helpers
    def change_direction(self, direction):
        random_int = random.randint(1,5)
        if random_int == 4: #left
            if direction == "down" or direction == "up":
                direction = "left"
            elif direction == "left":
                direction = "down"
            else:
                direction = "up"
        else: #right
            if direction == "down" or direction == "up":
                direction = "right"
            elif direction == "left":
                direction = "up"
            else:
                direction = "down"

        return direction
    def generate_river_start(self):
        random_block = self.random_block_on_boundary()
        self.set_block(random_block, self.river)
        r,c = random_block[0], random_block[1]

        if r == 0:
            direction = "down"
        elif r == self.max_row:
            direction = "up"
        elif c == 0:
            direction = "right"
        else:
            direction = "left"
        return (r,c, direction)
    def move_in_direction(self, r, c, direction, distance):
        dist = 0
        for i in range(distance):
            if direction == "down":
                r = r+1
            elif direction == "up":
                r = r-1
            elif direction == "right":
                c = c+1
            elif direction == "left":
                c = c-1

            if self.in_range((r,c)):
                if self.grid[r][c] >= self.hard_river: #we ran into ourselves
                    return (r, c, dist)
                else:
                    if self.grid[r][c] == self.hard:
                        self.set_block((r, c), self.hard_river)
                    else:
                        self.set_block((r, c), self.river)
                    dist += 1
            else:
                return (r, c, dist) #out of range

        return (r, c, dist)
    def random_block_on_boundary(self):
        random_int = random.randint(0,3)
        random_block = self.random_block()
        if random_int == 0:
            return (0, random_block[1])
        elif random_int == 1:
            return (self.max_row, random_block[1])
        elif random_int == 2:
            return (random_block[0], 0)
        else:
            return (random_block[0], self.max_col)

    def show_map(self):
        GREEN = 'green'
        WHITE = 'white' #unblocked
        LIGHT_GREY = '0.65' #hard
        BLUE_GREY= '#73C2E6'#hard_river
        BLUE = 'blue' #river
        DARK_GREY = '0.25' #blocked
        BLACK = 'red' #traversed

        import matplotlib.patches as mpatches

        cmap = mpl.colors.ListedColormap([GREEN, WHITE, LIGHT_GREY, BLUE_GREY, BLUE, DARK_GREY, BLACK])
        norm = mpl.colors.BoundaryNorm(self.bounds, cmap.N)
        img = plt.imshow(self.grid,
                            interpolation='nearest',
                            cmap = cmap,
                            norm = norm)

        values = [self.empty_goal, self.empty, self.hard, self.hard_river, self.river, self.blocked, self.traversed]
        value_strings = ["goal", "empty", "partial block", "hard + river", "river", "blocked", "traversed"]
        colors = [img.cmap(img.norm(value)) for value in values]

        block_types = [mpatches.Patch(color=colors[i], label="{s}".format(s=value_strings[i])) for i in range(len(value_strings))]
        hard_centers = [mpatches.Patch(label = "{s}".format(s=str(center))) for center in self.hard_centers]

        l1 = plt.legend(handles=block_types, bbox_to_anchor=(-.29, 1.1), loc=2, borderaxespad=0., title="Block types")
        l2 = plt.legend(handles=hard_centers, bbox_to_anchor=(1.03, 1.1), loc=2, borderaxespad=0., title="Centers")

        if self.start and self.goal:
            start_goal = [mpatches.Patch(label="start\n" + str(self.start)), mpatches.Patch(label="goal\n" + str(self.goal))]
            l3 = plt.legend(handles=start_goal, bbox_to_anchor=(1.03, .64), loc=2, borderaxespad=0., title="Start/Goal")

        plt.gca().add_artist(l1)
        plt.gca().add_artist(l2)

        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()

    # sets back to map without start/goal when called outside map
    # internally (inside map.py) it sets back to whatever self.grid_copy is
    def reset_map(self):
        self.grid = self.grid_copy
        self.grid_copy = np.copy(self.grid)

    def get_copy(self):
        return np.copy(self.grid)

    #helpers
    def get_neighbors(self, block, radius=1):
        row = block[0]
        col = block[1]
        r_max = row + radius
        neighbors = []
        r_min = row - radius
        c_max = col + radius
        c_min = col - radius
        for r in range(r_min,r_max+1):
            for c in range(c_min,c_max+1):
                if not (r==row and c==col) and self.in_range((r,c)):
                        neighbors.append((r,c))
        return neighbors

    def in_range(self, block):
        row, col = block[0], block[1]
        return (row>=self.min_row and row<=self.max_row) and (col>=self.min_col and col<=self.max_col)

    def set_block(self, block, value):
        row,col = block[0], block[1]
        self.grid[row][col] = value

    def get_block(self, block):
        return self.grid[block[0],block[1]]

    def manhattan_distance(self, *blocks):
        ans = 0
        n = len(blocks)-1
        for i in range(n):
            block1 = blocks[i]
            block2 = blocks[i+1]
            row1, col1 = block1[0], block1[1]
            row2, col2 = block2[0], block2[1]
            ans += abs(row1-row2) + abs(col1-col2)
        return ans

    def euclidean_distance(self, *blocks):
        ans = 0
        n = len(blocks)-1
        for i in range(n):
            block1 = blocks[i]
            block2 = blocks[i+1]
            row1, col1 = block1[0], block1[1]
            row2, col2 = block2[0], block2[1]
            ans += sqrt( (row1-row2) **2 + (col1 - col2) ** 2 )
        return ans

    def diagonal(self, block1, block2):
        row1, col1 = block1[0], block1[1]
        row2, col2 = block2[0], block2[1]
        d_row = abs(row2-row1)
        d_col = abs(col2-col1)
        minimum = min(d_row, d_col)
        maximum = max(d_row, d_col)

        diagonal = minimum
        straight = maximum - minimum
        return (sqrt(2)* diagonal) + straight
    def straight(self, block1, block2):
        heuristic = self.euclidean_distance(block1,block2)
        row0, col0 = self.start
        row1, col1 = block1[0], block1[1]
        row2, col2 = block2[0], block2[1]

        dr1 = row0-row2
        dc1 = col0-col2
        dr2 = row1-row2
        dc2 = col1-col2

        cross = abs(dr2*dc1 - dr1*dc2)
        heuristic += cross*0.001
        return heuristic

    def admissable(self, block1, block2):
        return .25 * self.manhattan_distance(block1, block2)


    #random
    def random_block(self, r_min=None, r_max=None, c_min=None, c_max=None, max_threshold = 100):
        # returns a random block within boundaries specified
        # boundaries defaults to the entire map
        # only blocks less than max_threshold can be returned
        if r_max==None:
            r_max = self.max_row
        if c_max==None:
            c_max = self.max_col
        if r_min==None:
            r_min = 0
        if c_min==None:
            c_min = 0

        rand_row = random.randint(r_min, r_max)
        rand_col = random.randint(c_min, c_max)

        while self.grid[rand_row][rand_col] > max_threshold:
            rand_row = random.randint(r_min, r_max)
            rand_col = random.randint(c_min, c_max)

        rand_block = (rand_row, rand_col)
        return rand_block
