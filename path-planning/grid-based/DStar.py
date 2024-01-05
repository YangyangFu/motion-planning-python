""" DStar algorithm
"""
import matplotlib.pyplot as plt

from env import Env 
from plotting import Plotting

class DStar:
    
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal

        # algorithm related
        self.t = {}
        self.h = {}
        self.k = {}
        self.path = []
        self.visited = set()
        self.count = 0
        
        self.open = set()
        self.parent = {}
        
        # map related
        self.env = Env()
        self.motions = self.env.motions
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obstacles = self.env.obs
        
        # visualization
        self.fig = plt.figure()
        self.plot = Plotting(self.start, self.goal)
        self.count = 0
    
    def initialize(self):
        for x in range(self.x_range):
            for y in range(self.y_range):
                self.t[(x, y)] = 'NEW'
                self.h[(x, y)] = float("inf")
                self.k[(x, y)] = 0
                self.parent[(x, y)] = None
        
        self.h[self.goal] = 0    
    
    def search(self):
        """Runs D* algorithm
        """
        # initalize algorithm
        self.initialize()
        # insert the goal into the OPEN list
        self.insert(self.goal, 0)
        
        # repeatly call process_state() until the start node is CLOSED    
        while True:
            self.process_state()
            
            if self.t[self.start] == 'CLOSED':
                break
            
            # robot moves one step forward following the path
            # if reach the goal then return
            # if detect a new obstacle then modify the cost and recompute the shortest path
        # extract the optimal path 
        path = self.extract_path(self.start, self.goal)
        
        # plot
        self.plot.plot_grid("D*")
        self.plot_path(path)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        plt.show()
            
    def process_state(self):
        """Calculates the optimal path costs to the goal 
            and returns the minimum key value in the OPEN list
        """
        # remove the state s with the smallest key from the OPEN list
        X = self.min_state()
        self.visited.add(X)
        if X is None:
            return -1

        # get min key value 
        k_old = self.get_kmin()
        # move state X from the OPEN list to the CLOSED list
        self.delete(X)
        
        if k_old < self.h[X]:
            for neighbor in self.get_neighbor(X):
                if self.h[neighbor] <= k_old and self.h[X] > self.h[neighbor] + self.distance(X, neighbor):
                    self.parent[X] = neighbor
                    self.h[X] = self.h[neighbor] + self.distance(X, neighbor)
        
        if k_old == self.h[X]:
            for neighbor in self.get_neighbor(X):
                # insert if 
                # 1. the state is NEW, not visited
                # 2. cost reduction: neighbor cost is not equal to the parent cost + distance
                # 3. neighbor find a better parent
                if self.t[neighbor] == 'NEW' \
                    or (self.parent[neighbor] == X and self.h[neighbor] != self.h[X] + self.distance(X, neighbor)) \
                    or (self.parent[neighbor] != X and self.h[neighbor] > self.h[X] + self.distance(X, neighbor)):
                    self.parent[neighbor] = X
                    self.insert(neighbor, self.h[X] + self.distance(X, neighbor))
        else:
            for neighbor in self.get_neighbor(X):
                # insert if 
                # 1. the state is NEW, not visited
                # 2. cost reduction after last time: neighbor cost is not equal to the parent cost + distance
                if self.t[neighbor] == 'NEW' \
                    or (self.parent[neighbor] == X and self.h[neighbor] != self.h[X] + self.distance(X, neighbor)):
                    self.parent[neighbor] = X
                    self.insert(neighbor, self.h[X] + self.distance(X, neighbor))
                else:
                    if self.parent[neighbor] != X and self.h[neighbor] > self.h[X] + self.distance(X, neighbor):
                        self.insert(X, self.h[X])
                    else:
                        if self.parent[neighbor] != X \
                            and self.h[X] > self.h[neighbor] + self.distance(X, neighbor) \
                            and self.t[neighbor] == 'CLOSED' \
                            and self.h[neighbor] > k_old:
                            self.insert(neighbor, self.h[neighbor])
        
        return self.get_kmin()                    

    def modify_cost(self, node, parent):
        """Modifies the edge costs of the graph, and insert the nodes into the OPEN list
        """
        if self.t[node] == 'CLOSED':
            self.insert(node, self.h[parent] + self.distance(parent, node))
            
    def min_state(self):
        """Returns the state on the OPEN list with the minimum key value. Null if the list is empty.
        """
        if not self.open:
            return None

        return min(self.open, key=lambda x: self.k[x])
    
    def get_kmin(self):
        """Returns the minimum key value in the OPEN list. return -1 if the list is empty. 
        """
        if not self.open:
            return -1
        
        return self.k[self.min_state()]
        
    def delete(self, node):
        """Deletes a state from the OPEN list, and set tag of node to be CLOSED. 
        """
        self.open.remove(node)
        self.t[node] = 'CLOSED'
        
    def insert(self, node, h_new):
        """Inserts a state into the OPEN list. 
        """
        if self.t[node] == 'NEW':
            self.k[node] = h_new
        elif self.t[node] == 'OPEN':
            self.k[node] = min(self.k[node], h_new)
        elif self.t[node] == 'CLOSED':
            self.k[node] = min(self.h[node], h_new)
        
        self.h[node] = h_new
        self.t[node] = 'OPEN'
        self.open.add(node)
    
    def get_neighbor(self, node):
        """ Get the neighbors of a node that is not in the obstacles"""
        neighbors = []
        for motion in self.motions:
            x = node[0] + motion[0]
            y = node[1] + motion[1]
            if (x,y) not in self.obstacles and x >= 0 and x < self.env.x_range and y >= 0 and y < self.env.y_range:
                neighbors.append((x, y))
        return neighbors
    
    def distance(self, node1, node2):
        if self.has_collision(node1, node2):
            return float("inf")
        
        return ((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5

    def extract_path(self, start, goal):
        path = [start]
        node = start
        while True:
            node = self.parent[node]
            path.append(node)
            if node == goal:
                break
        return path
    
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

            self.visited = set()
            self.count += 1

            if (x, y) not in self.obstacles:
                self.obstacles.add((x, y))
            else:
                self.obstacles.remove((x, y))


            self.plot.update_obs(self.obstacles)

            node = self.start
            while node != self.goal:
                if self.has_collision(node, self.parent[node]):
                    self.modify_cost(node, self.parent[node])
                    while True:
                        kmin = self.process_state()
                        if kmin >= self.h[node]:
                            break
                    continue
                
                node = self.parent[node]

            path = self.extract_path(self.start, self.goal)

            plt.cla()
            self.plot.plot_grid("LPA*")
            self.plot_visited(self.visited)
            self.plot_path(path)
            self.fig.canvas.draw_idle()
            
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

if __name__ == "__main__":
    start = (5, 5)
    goal = (45, 25)

    dstar = DStar(start, goal)
    dstar.search()
    