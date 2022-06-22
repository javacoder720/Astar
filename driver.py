import matplotlib.pyplot as plt
import pickle
import pandas as pd

from map import Map
from agent import Agent

class Driver:
    def __init__(self):
        self.show = False
        self.maps = []
        self.start_goal_pairs = []

        self.rows = ["map", "start/goal", "heuristic", "weight"]
        self.rows2 = ["map", "start/goal", "heuristics", "weight","weight2"]
        self.columns = ["iterations", "distance", "time", "memory"]
        self.heuristics = ["admissable", "manhattan", "euclidean", "diagonal","straight"]

        self.weights0 = [0]
        self.weights = [1, 1.125, 1.25, 1.375, 1.5]
        self.weights2 = [1,1.5,2]
        self.weights3 = [1.5,2,2.5]

        self.metric = None

    def run_agents(self, show=False):
        self.show=show
        self.create_indexes()

        for i in range(len(self.maps)):
            print("map: {}".format(i))
            for j, pair in enumerate(self.start_goal_pairs[i]):
                print("pair: {}".format(pair))
                start, goal = pair
                self.do_search(i, j, start, goal, "manhattan", 0)
                for weight in self.weights:
                    for heuristic in self.heuristics:
                        self.do_search(i, j, start, goal, heuristic, weight)
        self.show_results(levels=["heuristic"])
        # self.show_results(levels=["weight"])
    def run_sequential_agents(self, show=False):
        self.show=show
        self.create_sequential_indexes()
        for i in range(len(self.maps)):
            print("map: {}".format(i))
            for j, pair in enumerate(self.start_goal_pairs[i]):
                print("pair: {}".format(pair))
                for weight2 in self.weights2:
                    for weight3 in self.weights3:
                        start, goal = pair
                        self.do_sequential_search(i, j, start, goal, weight2, weight3)


        self.show_results(levels=["weight","weight2"])

    def do_search(self, i, j, start, goal, heuristic, weight):
        map = self.maps[i]
        map.set_name("map_{}".format(i))
        map.set_start_and_goal(start, goal)
        agent = Agent(start, goal, map, self.show)

        index = self.get_index(i,start,goal,heuristic,weight)
        metric = agent.search(heuristic, weight=weight)
        if weight==0:
            self.optimal_distance = metric[1]
        metric = (metric[0], metric[1]/self.optimal_distance, metric[2], metric[3])

        self.save_metrics(index, metric)

    def do_sequential_search(self, i, j, start, goal, weight, weight2):
        map = self.maps[i]
        map.set_name("map_{}".format(i))
        map.set_start_and_goal(start, goal)
        agent = Agent(start, goal, map, self.show)

        index = self.get_sequential_index(i,start,goal,weight,weight2)
        metric = agent.sequential_search(self.heuristics, weight=weight, weight2=weight2)
        self.save_metrics(index, metric)

    def save_metrics(self, info, metric):
        for column,metric in zip(self.columns, metric):
            self.df.loc[info, column] = metric

    def save_maps(self):
        with open("maps" + '.pkl', 'wb') as f:
            pickle.dump(self.maps, f)
        with open("start_goal_pairs" + '.pkl', 'wb') as f:
            pickle.dump(self.start_goal_pairs, f)

    def generate_maps(self, num_maps, num_start_goal):
        self.num_maps = num_maps
        self.num_start_goal = num_start_goal
        size = (120, 160)
        for i in range(self.num_maps):
            map = Map(size)
            self.maps.append(map)
            pairs = []
            for j in range(self.num_start_goal):
                map.generate_start_and_end()
                start, goal = map.start, map.goal
                pairs.append( (start,goal) )
                map.reset_map()
            self.start_goal_pairs.append(pairs)

    def load_maps(self, maps="maps", start_goal_pairs="start_goal_pairs"):
        with open(maps + '.pkl', 'rb') as f:
            self.maps = pickle.load(f)
        with open(start_goal_pairs + '.pkl', 'rb') as f:
            self.start_goal_pairs = pickle.load(f)

    def show_maps(self):
        for i in range(self.num_maps):
            map = self.maps[i]
            map.show_map()
            plt.show()
            
    def show_maps_and_pairs(self):
        for i in range(self.num_maps):
            map = self.maps[i]
            for pair in self.start_goal_pairs[i]:
                start,goal = pair
                map.set_start_and_goal(start, goal)
                map.show_map()
                plt.show()
                map.reset_map()

    def show_results(self, levels, max_col_width=20):
        self.df['iterations'] = self.df['iterations'].astype(float)
        self.df['distance'] = self.df['distance'].astype(float)
        self.df['time'] = self.df['time'].astype(float)
        self.df['memory'] = self.df['memory'].astype(int)

        pd.set_option('display.max_rows', len(self.df))
        pd.set_option('max_colwidth', max_col_width)
        print(self.df)
        weights = self.df.groupby(levels)
        # print(weights.mean().sort_values(["iterations"]))
        # print(weights.median().sort_values(["iterations"]))
        print(weights.mean().sort_values(["distance"]))
        print(weights.median().sort_values(["distance"]))

    def create_indexes(self):
        indexes = []

        for i in range(len(self.maps)):
            for j, pair in enumerate(self.start_goal_pairs[i]):
                start, goal = pair
                indexes.append( (i, "{} -> {}".format(start, goal), "manhattan", 0) )
                for weight in self.weights:
                    for heuristic in self.heuristics:
                        indexes.append( (i, "{} -> {}".format(start, goal), heuristic, weight) )

        index = pd.MultiIndex.from_tuples(indexes, names=self.rows)
        self.df = pd.DataFrame(index=index, columns=self.columns)
    def create_sequential_indexes(self):
        indexes = []

        for i in range(len(self.maps)):
            for j, pair in enumerate(self.start_goal_pairs[i]):
                    for weight2 in self.weights2:
                        for weight3 in self.weights3:
                            start,goal = pair
                            indexes.append(self.get_sequential_index(i, start,goal,weight2,weight3))

        index = pd.MultiIndex.from_tuples(indexes, names=self.rows2)
        self.df = pd.DataFrame(index=index, columns=self.columns)

    def get_index(self,i,start,goal,heuristic,weight):
        return [(i, "{} -> {}".format(start, goal), heuristic, weight)]
    def get_sequential_index(self,i,start,goal,weight,weight2):
        heuristics = ", ".join(self.heuristics)
        index = (i, "{} -> {}".format(start, goal), heuristics, weight, weight2)
        return index
