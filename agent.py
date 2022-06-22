import heapq
import math
from state import State
from map import Map
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, start, goal, map, show):
        self.map = map
        self.show = show

        self.start = start
        self.goal = goal

        if show:
            fig = plt.figure()
            cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
    def reset(self):
        from collections import defaultdict
        self.open_lists = defaultdict(lambda: [])
        self.open_lists_hash = defaultdict(lambda: {})
        self.closed_lists = defaultdict(lambda: {})

        self.max_memories = defaultdict(lambda: 0)
        self.iterations = defaultdict(lambda: 0)
        self.metrics = {}

        self.current_states = {}
        self.goal_states = {}

    def search(self, heuristic, weight):
        self.reset()
        goal_state = State(self.goal)
        self.goal_states[heuristic] = goal_state
        current_state = State(self.start)
        current_state.set_f(0, self.calculate_h(current_state, heuristic, weight))
        heapq.heappush(self.open_lists[heuristic], current_state)

        import timeit
        time = timeit.default_timer()
        while True:
            self.compute_path(heuristic, weight)
            goal_location = self.goal_states[heuristic].location
            if self.current_states[heuristic].location == goal_location:
                break
            if len(self.open_lists[heuristic]) == 0:
                break

        self.selected_heuristic = heuristic
        if self.show:
            title = "map: {}\nweight: {}\nheuristic: {}".format(self.map.get_name(), weight, heuristic)
            self.show_map(title)
        self.map.reset_map()

        iterations = self.iterations[heuristic]
        distance = self.goal_states[heuristic].g
        time = timeit.default_timer() - time
        memory = self.max_memories[heuristic]
        metric = (iterations, distance, time, memory)

        return metric

    def sequential_search(self, heuristics, weight, weight2):
        self.reset()
        import timeit
        time = timeit.default_timer()

        for i, heuristic in enumerate(heuristics):
            goal = State(self.goal)
            self.goal_states[heuristic] = goal

            start = State(self.start)
            start.set_f(0, self.calculate_h(start, heuristic, weight))
            self.open_lists[heuristic].append(start)

        heuristic = self.search_loop(heuristics, weight, weight2)
        self.selected_heuristic = heuristic

        if self.show:
            title = "map: {}\nweight: {}, weight2: {}\nheuristics: {}".format(self.map.get_name(), weight, weight2, heuristics)
            self.show_map(title)
        self.map.reset_map()

        iterations = self.iterations[heuristic]
        distance = self.goal_states[heuristic].g
        time = timeit.default_timer() - time
        memory = self.max_memories[heuristic]
        metric = (iterations, distance, time, memory)

        return  metric
    def search_loop(self, heuristics, weight, weight2):
        while True:
            for i in range(1,len(heuristics)):
                h_0 = heuristics[0]
                s_0 = self.open_lists[h_0][0]
                goal_0 = self.goal_states[h_0]
                h_i = heuristics[i]
                s_i = self.open_lists[h_i][0]
                goal_i = self.goal_states[h_i]

                if s_i.f <= weight2 * s_0.f:
                    if goal_i.g <= s_i.f:
                        if goal_i.g < math.inf:
                            return h_i
                    else:
                        self.compute_path(h_i, weight)
                else:
                    if goal_0.g <= s_0.f:
                        if goal_0.g < math.inf:
                            return h_0
                    else:
                        self.compute_path(h_0, weight)

    def compute_path(self, heuristic, weight):
        self.iterations[heuristic] += 1
        open_list = self.open_lists[heuristic]
        self.current_states[heuristic] = heapq.heappop(open_list)
        state = self.current_states[heuristic]

        location = state.location
        if location in self.closed_lists[heuristic]:
            return

        goal = self.goal_states[heuristic]
        goal_location = goal.location

        if location == goal_location:
            new_g = self.calculate_g(state, goal)
            goal.set_g(new_g)
            goal.set_parent(state.parent)
            return

        self.closed_lists[heuristic][location] = state
        neighbors = self.map.get_neighbors(location)
        self.procces_neighbors(neighbors, heuristic, weight)
        self.max_memories[heuristic] = max(self.max_memories[heuristic], len(open_list))

    def procces_neighbors(self, neighbors, heuristic, weight):
        for neighbor in neighbors:
            if neighbor in self.closed_lists[heuristic]:
                continue

            if self.map.get_block(neighbor) > self.map.river:
                continue

            state = self.get_from_open_list(neighbor, heuristic)
            self.update_in_open_list(state, heuristic, weight)

    def get_from_open_list(self, location, heuristic):
        open_list_hash = self.open_lists_hash[heuristic]

        if location in open_list_hash:
            return open_list_hash[location]
        else:
            return State(location)

    def update_in_open_list(self, state, heuristic, weight):
        open_list = self.open_lists[heuristic]
        open_list_hash = self.open_lists_hash[heuristic]
        current_state = self.current_states[heuristic]

        if state.parent is None:
            state.set_parent(current_state)

        parent = state.parent

        new_g = self.calculate_g(parent, state);
        if state.g > new_g:
            new_h = self.calculate_h(state, heuristic, weight)
            state.set_f(new_g,  new_h)
            state.set_parent(current_state)

            heapq.heappush(open_list, state)
            open_list_hash[state.location] = state


    def calculate_h(self, state, heuristic, weight):
        location = state.location
        location_goal = self.goal_states[heuristic].location

        if heuristic == "admissable":
            return weight * self.map.admissable(location,location_goal)
        elif heuristic == "manhattan":
            return weight * self.map.manhattan_distance(location, location_goal)
        elif heuristic == "euclidean":
            return weight * self.map.euclidean_distance(location, location_goal)
        elif heuristic == "diagonal":
            return weight * self.map.diagonal(location, location_goal)
        elif heuristic == "straight":
            return weight * self.map.straight(location, location_goal)

    def calculate_g(self, parent, neighbor):
        p_location = parent.location
        g = parent.g
        n_location = neighbor.location
        p_block = self.map.get_block(p_location)
        n_block = self.map.get_block(n_location)

        p_hard = p_block == self.map.hard
        n_hard = n_block == self.map.hard
        p_river = p_block == self.map.river
        n_river = n_block == self.map.river

        both_hard = p_hard and n_hard
        one_hard = p_hard ^ n_hard
        both_river = p_river and n_river

        diagonal = False
        if self.map.manhattan_distance(p_location, n_location) == 2:
            diagonal = True

        from math import sqrt
        if both_hard:
            if diagonal:
                g += sqrt(8)
            else:
                g += 2
        elif one_hard:
            if diagonal:
                g += .5 * (sqrt(2) + sqrt(8))
            else:
                g += 1.5
        else:
            if diagonal:
                g += sqrt(2)
            else:
                g += 1

        if both_river:
            g *= (.25)

        return g

    def show_map(self, title):
        if self.show:
            plt.title(title)
            self.color_map()
            self.map.show_map()
    def color_map(self):
        h = self.selected_heuristic
        state = self.goal_states[h]
        while state.parent != None:
            self.map.set_block(state.location, self.map.traversed)
            state = state.parent
    def onclick(self, event):
        h = self.selected_heuristic
        r, c = int(event.ydata), int(event.xdata)
        print ('({}, {})'.format(r,c))
        if (r,c) in self.closed_lists[h]:
            state = self.closed_lists[h][(r,c)]
            g = state.g
            f = state.h
            h = state.h
            block = 'g={}\nh={}\nf={}'.format(g, h, f)
            print(block)
