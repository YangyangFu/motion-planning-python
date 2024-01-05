import matplotlib.pyplot as plt

from env import Env
from plotting import Plotting 

class DStarLite():
    def __init__(self, start, goal, heuristic_type):
        self.start = start
        self.goal = goal
        self.heuristic_type = heuristic_type
        
        # map related 
        self.env = Env()
        self.motions = self.env.motions
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obstacles = self.env.obs
        
        # algorithm related
        self.k_m = 0
        self.g_values = dict()
        self.rhs_values = dict()
        self.open = dict()
        
        # visualization
        self.fig = plt.figure()
        self.plot = Plotting(self.start, self.goal)
        self.count = 0        
        
        # initialize
        self.initialize()
        
    def initialize(self):
        """ Initialize g and rhs values for all nodes
        """ 
        for x in range(self.x_range):
            for y in range(self.y_range):
                self.g_values[(x, y)] = float("inf")
                self.rhs_values[(x, y)] = float("inf")
        
        self.g_values[self.goal] = float("inf")
        self.rhs_values[self.goal] = 0
        self.open[self.goal] = self.calculate_key(self.goal)
    
    def search(self):
        
        self.plot.plot_grid("D* Lite")
    
        self.compute_shortest_path()
        path = self.extract_path()
        self.plot_visited(self.visited)
        self.plot_path(path) 
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        plt.show()
    
    
    def on_press(self, event):
        """Event handler for key press event
        
        If a key is pressed, the position of the mouse is assumed to be a new obstacle. 
        - new obstacle on the path detected
        - update the node with new distance 
        - compute the shortest path
        - update the plot
        """
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.env.x_range - 1 or y < 0 or y > self.env.y_range - 1:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            print("Change position: s =", x, ",", "y =", y)

            self.count += 1

            if (x, y) not in self.obstacles:
                self.obstacles.add((x, y))
            else:
                self.obstacles.remove((x, y))
                self.update_node((x, y))

            self.plot.update_obs(self.obstacles)

            for s_n in self.get_neighbors((x, y)):
                self.update_node(s_n)

            self.compute_shortest_path()

            plt.cla()
            self.plot.plot_grid("D* Lite")
            self.plot_visited(self.visited)
            path = self.extract_path()
            self.plot_path(path)
            self.fig.canvas.draw_idle()        
    
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
            if neighbor[0] < 0 or neighbor[0] >= self.x_range or neighbor[1] < 0 or neighbor[1] >= self.y_range:
                continue
            if neighbor in self.obstacles:
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
        # for visualization only 
        self.visited = set()
        
        # main loop
        while self.top_key()[1] < self.calculate_key(self.start) or self.rhs_values[self.start] > self.g_values[self.start]:
            k_old = self.top_key()[1]
            u = self.top_key()[0]
            k_new = self.calculate_key(u)
            
            self.open.pop(u)
            self.visited.add(u)
            
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
        
        if node1 in self.obstacles or node2 in self.obstacles:
            return True

        if node1[0] != node2[0] and node1[1] != node2[1]:
            if node2[0] - node1[0] == node1[1] - node2[1]:
                s1 = (min(node1[0], node2[0]), min(node1[1], node2[1]))
                s2 = (max(node1[0], node2[0]), max(node1[1], node2[1]))
            else:
                s1 = (min(node1[0], node2[0]), max(node1[1], node2[1]))
                s2 = (max(node1[0], node2[0]), min(node1[1], node2[1]))

            if s1 in self.obstacles or s2 in self.obstacles:
                return True

        return False

    def plot_path(self, path):
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2)
        plt.plot(self.start[0], self.start[1], "bs")
        plt.plot(self.goal[0], self.goal[1], "gs")

    def plot_visited(self, visited):
        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        if self.count >= len(color) - 1:
            self.count = 0

        for x in visited:
            plt.plot(x[0], x[1], marker='s', color=color[self.count])

if __name__ == '__main__':
    x_start = (5, 5)
    x_goal = (45, 25)

    dstar_lite = DStarLite(x_start, x_goal, "euclidean")
    dstar_lite.search()