""" DStar algorithm
"""
import matplotlib.pyplot as plt
import cv2 as cv 

from env import Map 

class DStar:
    """D* algorithm
    """
    
    def __init__(self, start, goal):
        self.name = "D*"
        self.start = start
        self.goal = goal

        # algorithm related
        self.t = {}
        self.h = {}
        self.k = {}
        self.path = []
        self.visited = []        
        self.open = set()
        self.parent = {}
        
        # map related
        self.env = Map(title=self.name, type="ground_truth")
        self.env.set_start(start)
        self.env.set_goal(goal)
        self.motions = self.env.motions
        self.obstacles = self.env.obstacles
        
        # visualization
        self.count = 0
        
        # initialize
        self.initialize()
    
    def initialize(self):
        for x in range(self.env.x_lim):
            for y in range(self.env.y_lim):
                self.t[(x, y)] = 'NEW'
                self.h[(x, y)] = float("inf")
                self.k[(x, y)] = 0
                self.parent[(x, y)] = None
        
        self.h[self.goal] = 0    
    
    def search(self):
        """Runs D* algorithm
        """
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
        
        return path
            
    def process_state(self):
        """Calculates the optimal path costs to the goal 
            and returns the minimum key value in the OPEN list
        """
        # remove the state s with the smallest key from the OPEN list
        X = self.min_state()
        self.visited.append(X)
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
            if (x,y) not in self.obstacles and x >= 0 and x < self.env.x_lim and y >= 0 and y < self.env.y_lim:
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
            
            # plot updated path and visited nodes
            self.env.plot_visited(self.visited)
            self.env.plot_path(path)
            
            self.count += 1
            print("udpate after click ", self.count)
            # show update
            cv.imshow(self.name, self.env.grid)

if __name__ == "__main__":
    import cv2 as cv
    
    start = (5, 5)
    goal = (25, 45)

    dstar = DStar(start, goal)
    dstar.plot() 
    path = dstar.search()
    dstar.env.plot_visited(dstar.visited)
    dstar.env.plot_path(path)
    
    cv.waitKey(0)
    
    dstar.env.save_gif(name='dstar.gif', duration=20)