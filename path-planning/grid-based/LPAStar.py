import heapq 
import matplotlib.pyplot as plt

from env import Env 
from plotting import Plotting

class LPAStar():
    """Life-long Planning A* algorithm
    """
    def __init__(self, start, goal, heuristic_type):
        self.start = start
        self.goal = goal
        self.heuristic_type = heuristic_type
        
        self.env = Env()
        self.motions = self.env.motions
        
        # map related 
        self.env = Env()
        self.motions = self.env.motions # feasible moving directions
        self.obs = self.env.obs # observation
        
        # initialize
        self.g_values = dict() 
        self.rhs_values = dict()
        self.open = [] # priority queue
        
        # visualization
        self.fig = plt.figure()
        