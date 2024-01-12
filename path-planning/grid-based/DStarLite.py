import matplotlib.pyplot as plt
import cv2 as cv 

from env import Map

class DStarLite():
    """ D* Lite algorithm
    """
    def __init__(self, start, goal, heuristic_type):
        self.name = "D* Lite"
        self.start = start
        self.goal = goal
        self.heuristic_type = heuristic_type
        
        # map related 
        self.env = Map(title=self.name, type="ground_truth")
        self.env.set_start(start)
        self.env.set_goal(goal)
        self.motions = self.env.motions
        
        # algorithm related
        self.k_m = 0
        self.g_values = dict()
        self.rhs_values = dict()
        self.open = dict()
        
        # visualization
        self.count = 0        
        self.visited = [] 
        
        # initialize
        self.initialize()
        
    def initialize(self):
        """ Initialize g and rhs values for all nodes
        """ 
        for x in range(self.env.x_lim):
            for y in range(self.env.y_lim):
                self.g_values[(x, y)] = float("inf")
                self.rhs_values[(x, y)] = float("inf")
        
        self.g_values[self.goal] = float("inf")
        self.rhs_values[self.goal] = 0
        self.open[self.goal] = self.calculate_key(self.goal)
    
    def search(self):
        
        self.compute_shortest_path()
        path = self.extract_path()

        return path

    def heuristic(self, node):
        """ Calculate the heuristic value of a node

        Args:
            node (_type_): _description_
        """
        if self.heuristic_type == "manhattan":
            return abs(node[0] - self.start[0]) + abs(node[1] - self.start[1])
        elif self.heuristic_type == "euclidean":
            return ((node[0] - self.start[0]) ** 2 + (node[1] - self.start[1]) ** 2) ** 0.5
    
    def get_neighbors(self, node):
        """ Get all neighbors of a node

        Args:
            node (_type_): _description_

        Returns:
            _type_: _description_
        """
        neighbors = []
        for motion in self.motions:
            neighbor = (node[0] + motion[0], node[1] + motion[1])
            if neighbor[0] < 0 or neighbor[0] >= self.env.x_lim or neighbor[1] < 0 or neighbor[1] >= self.env.y_lim:
                continue
            if neighbor in self.env.obstacles:
                continue
            neighbors.append(neighbor)
        return neighbors
    
    def get_predecessors(self, node):
        return self.get_neighbors(node)
    
    def get_successors(self, node):
        return self.get_neighbors(node)
    
    def top_key(self):
        """ Get the top key in the priority queue

        Returns:
            
        """
        node = min(self.open, key=self.open.get)
        return node, self.open[node]
            
    def calculate_key(self, node):
        return (min(self.g_values[node], self.rhs_values[node]) + self.heuristic(node) + self.k_m, 
                min(self.g_values[node], self.rhs_values[node])
        ) 
    
    def update_node(self, node):
        """ Recalculate rhs for a node and removes it from queue.
        If the node has become locally inconsistent, it is reinserted into the queue with its new key

        Here assume that the graph is bidirectional where predecessor and successor are the same. 
        """
        if self.g_values[node] != self.rhs_values[node] and node in self.open:
            self.open[node] = self.calculate_key(node)
        elif self.g_values[node] != self.rhs_values[node] and node not in self.open:
            self.open[node] = self.calculate_key(node)
        elif self.g_values[node] == self.rhs_values[node] and node in self.open:
            self.open.pop(node)
        
    def compute_shortest_path(self):
        """Compute the shortest path from start to goal
        """
                
        # main loop
        while self.top_key()[1] < self.calculate_key(self.start) or self.rhs_values[self.start] > self.g_values[self.start]:
            k_old = self.top_key()[1]
            u = self.top_key()[0]
            k_new = self.calculate_key(u)
            
            self.open.pop(u)
            self.visited.append(u)
            
            if k_old < k_new:
                self.open[u] = k_new
            elif self.g_values[u] > self.rhs_values[u]:
                self.g_values[u] = self.rhs_values[u]
                for s in self.get_predecessors(u):
                    if s != self.goal:
                        self.rhs_values[s] = min(self.rhs_values[s], self.g_values[u] + self.distance(u, s))
                    self.update_node(s)
            else:
                g_old = self.g_values[u]
                self.g_values[u] = float("inf")
                for s in self.get_predecessors(u) + [u]:
                    if self.rhs_values[s] == self.distance(s, u) + g_old:
                        if s != self.goal:
                            self.rhs_values[s] = min(self.rhs_values[s], 
                                min([self.distance(s, s1) + self.g_values[s1] for s1 in self.get_successors(s)]))
                    self.update_node(s)

    def extract_path(self):
        node = self.start 
        path = [node]
        
        while node != self.goal:
            g_list = {}
            for x in self.get_neighbors(node):
                if not self.has_collision(node, x):
                    g_list[x] = self.g_values[x]
            node = min(g_list, key=g_list.get)
            path.append(node)
            
        return path
    
    def distance(self, node1, node2):
        """ Cost of an edge 
        """
        if self.has_collision(node1, node2):
            return float("inf")
        
        return ((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5

    def has_collision(self, node1, node2):
        
        if node1 in self.env.obstacles or node2 in self.env.obstacles:
            return True

        if node1[0] != node2[0] and node1[1] != node2[1]:
            if node2[0] - node1[0] == node1[1] - node2[1]:
                s1 = (min(node1[0], node2[0]), min(node1[1], node2[1]))
                s2 = (max(node1[0], node2[0]), max(node1[1], node2[1]))
            else:
                s1 = (min(node1[0], node2[0]), max(node1[1], node2[1]))
                s2 = (max(node1[0], node2[0]), min(node1[1], node2[1]))

            if s1 in self.env.obstacles or s2 in self.env.obstacles:
                return True

        return False

    def plot(self):
        """Plots the map
        """
        cv.namedWindow(self.name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.name, self.update_obstacles)
        self.env.draw_grid()
        cv.imshow(self.name, self.env.grid)
    
    
    def update_obstacles(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            row = int(y)
            col = int(x)

            # empty visited set
            self.visited = []
            
            # update obstacles
            if (row, col) not in self.env.obstacles:
                self.env.obstacles.add((row, col))
            else:
                self.env.obstacles.remove((row, col))
                # white out the grid
                self.env.grid[row, col] = [255, 255, 255]
            
            # plot obstacles
            for obstacle in self.env.obstacles:
                self.env.grid[obstacle[0], obstacle[1]] = [0, 0, 0]

            # save current frame for movie
            self.env.frames.append(self.env.grid.copy())
            
            # replot the grid with dynamic obstacles
            self.plot()
            
            # research the path
            path = self.search()

            # plot updated path and visited nodes
            self.env.plot_visited(self.visited)
            self.env.plot_path(path)
            
            self.count += 1
            print("udpate after click ", self.count)
            # show update
            cv.imshow(self.name, self.env.grid)

if __name__ == '__main__':
    x_start = (5, 5)
    x_goal = (25, 45)

    dstar_lite = DStarLite(x_start, x_goal, "euclidean")
    dstar_lite.plot()
    cv.waitKey(0)
     
    path = dstar_lite.search()
    dstar_lite.env.plot_visited(dstar_lite.visited)
    dstar_lite.env.plot_path(path)
    cv.waitKey(0)
    
    #dstar_lite.env.save_gif(name='dstarlite.gif', duration=20)
    cv.destroyAllWindows()