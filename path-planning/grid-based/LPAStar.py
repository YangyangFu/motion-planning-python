import matplotlib.pyplot as plt
import cv2 as cv 

from env import Map 

class LPAStar():
    """Life-long Planning A* algorithm
    """
    def __init__(self, start, goal, heuristic_type):
        self.name = "LPA*"
        self.start = start
        self.goal = goal
        self.heuristic_type = heuristic_type
                
        # map related 
        self.env = Map(title=self.name, type="ground_truth")
        self.env.set_start(start)
        self.env.set_goal(goal)
        self.motions = self.env.motions # feasible moving directions

        # initialize
        self.g_values = dict() 
        self.rhs_values = dict()
        self.open = dict() # priority queue {node: key}
        self.visited = []
        
        self.initialize()
        
        # visualization
        self.count = 0
        
    def initialize(self):
        """ Initialize g and rhs values for all nodes
        """ 
        for x in range(self.env.x_lim):
            for y in range(self.env.y_lim):
                self.g_values[(x, y)] = float("inf")
                self.rhs_values[(x, y)] = float("inf")
        
        self.g_values[self.start] = float("inf")
        self.rhs_values[self.start] = 0
        self.open[self.start] = self.calculate_key(self.start) # {node: key}
            
    def heuristic(self, node):
        """ Calculate the heuristic value of a node

        Args:
            node (_type_): _description_
        """
        if self.heuristic_type == "manhattan":
            return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])
        elif self.heuristic_type == "euclidean":
            return ((node[0] - self.goal[0]) ** 2 + (node[1] - self.goal[1]) ** 2) ** 0.5
    
    def update_node(self, node):
        """ Recalculate rhs for a node and removes it from queue.
        If the node has become locally inconsistent, it is reinserted into the queue with its new key
    
        Here assume that the graph is bidirectional where predecessor and successor are the same. 
        """
        # recalculate rhs value
        # if not start
        #   for predecessor in predecessors(node)
        #       node.rhs = min(node.rhs, predecessor.g + cost(predecessor, node))
        if node != self.start: 
            self.rhs_values[node] = min(self.g_values[predecessor] + self.distance(predecessor, node) 
                                        for predecessor in self.get_predecessor(node))
        
        # remove from queue: open: [(key, value)]
        if node in self.open:
            self.open.pop(node)
        
        # if locally inconsistent
        if self.g_values[node] != self.rhs_values[node]:
            self.open[node] = self.calculate_key(node)
    
    def calculate_key(self, node):
        return (min(self.g_values[node], self.rhs_values[node]) + self.heuristic(node), 
                min(self.g_values[node], self.rhs_values[node])
        )
    
    def distance(self, node1, node2):
        """ Cost of an edge 
        """
        if self.has_collision(node1, node2):
            return float("inf")
        
        return ((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5
        
    def get_predecessor(self, node):
        """ Get the predecessor of a node """
        predecessors = self.get_neighbor(node)
        return predecessors
    
    def get_successor(self, node):
        return self.get_neighbor(node)
    
    def get_neighbor(self, node):
        """ Get the neighbors of a node that is not in the obstacles"""
        neighbors = []
        for motion in self.motions:
            x = node[0] + motion[0]
            y = node[1] + motion[1]
            if (x,y) not in self.env.obstacles and x >= 0 and x < self.env.x_lim and y >= 0 and y < self.env.y_lim:
                neighbors.append((x, y))
        return neighbors
    
    def top_key(self):
        """ Get the top key of a priority queue 
        """
        node = min(self.open, key=self.open.get)
        
        return node, self.open[node]
    
    def compute_shortest_path(self):
        """ Compute the shortest path
        """
        while self.top_key()[1] < self.calculate_key(self.goal) or self.rhs_values[self.goal] != self.g_values[self.goal]:
            # pop 
            node, key = self.top_key()
            self.open.pop(node)
            
            # add to visited
            self.visited.append(node)
            
            # if g > rhs, overconsistent -> e.g. obstacle removed
            #  g = rhs
            if self.g_values[node] > self.rhs_values[node]:
                self.g_values[node] = self.rhs_values[node]
                # update successors due to g changes
                for successor in self.get_successor(node):
                    self.update_node(successor)
            # underconsistent -> e.g. obstacle added
            else:
                self.g_values[node] = float("inf")
    
                self.update_node(node)
                for successor in self.get_successor(node):
                    self.update_node(successor)
    
    def search(self, ):

        # compute the shortest path 
        self.compute_shortest_path()
        path = self.extract_path()

        return path

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.goal]
        node = self.goal

        for k in range(100):
            g_list = {}
            for neighbor in self.get_neighbor(node):
                if not self.has_collision(node, neighbor):
                    g_list[neighbor] = self.g_values[neighbor]
            node = min(g_list, key=g_list.get)
            path.append(node)
            if node == self.start:
                break

        return list(reversed(path))

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

    lpastar = LPAStar(x_start, x_goal, "euclidean")
    lpastar.plot()
    cv.waitKey(0)
     
    path = lpastar.search()
    lpastar.env.plot_visited(lpastar.visited)
    lpastar.env.plot_path(path)
    cv.waitKey(0)
    
    #lpastar.env.save_gif(name='lpastar.gif', duration=20)
    cv.destroyAllWindows()