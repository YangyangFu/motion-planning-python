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
                
        # map related 
        self.env = Env()
        self.motions = self.env.motions # feasible moving directions
        self.obs = self.env.obs # observation
        
        # initialize
        self.g_values = dict() 
        self.rhs_values = dict()
        self.open = [] # priority queue
        self.visited = set()
        
        self.initialize()
        
        # visualization
        self.fig = plt.figure()
        self.plot = Plotting(self.start, self.goal)
        self.count = 0
        
    def initialize(self):
        """ Initialize g and rhs values for all nodes
        """ 
        for x in range(self.env.x_range):
            for y in range(self.env.y_range):
                self.g_values[(x, y)] = float("inf")
                self.rhs_values[(x, y)] = float("inf")
        
        self.g_values[self.start] = float("inf")
        self.rhs_values[self.start] = 0
        self.open.append((self.calculate_key(self.start), self.start)) # [(key, value)]
            
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
        
        # remove from queue
        if node in self.open:
            self.open.remove(node)
        
        # if locally inconsistent
        if self.g_values[node] != self.rhs_values[node]:
            heapq.heappush(self.open, (self.calculate_key(node), node))
    
    def calculate_key(self, node):
        return (min(self.g_values[node], self.rhs_values[node]) + self.heuristic(node), 
                min(self.g_values[node], self.rhs_values[node])
        )
    
    def distance(self, node1, node2):
        """ Cost of an edge 
        """
        return ((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) ** 0.5
        
    def get_predecessor(self, node):
        """ Get the predecessor of a node """
        predecessors = self.get_neighbor(node)
        return predecessors
    
    def get_successor(self, node):
        return self.get_neighbor(node)
    
    def get_neighbor(self, node):
        """ Get the neighbors of a node """
        neighbors = []
        for motion in self.motions:
            x = node[0] + motion[0]
            y = node[1] + motion[1]
            neighbors.append((x, y))
        return neighbors
    
    def top_key(self):
        """ Get the top key of a priority queue 
        """
        heapq.heapify(self.open)
        
        return self.open[0][0] if self.open else (float("inf"), float("inf"))
    
    def compute_shortest_path(self):
        """ Compute the shortest path
        """
        
        while self.top_key() < self.calculate_key(self.goal) or self.rhs_values[self.goal] != self.g_values[self.goal]:
            # pop 
            key, node = heapq.heappop(self.open)
            # add to visited
            self.visited.add(node)
            
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
        self.plot.plot_grid("LPA*")
        # compute the shortest path 
        self.compute_shortest_path()
        path = self.extract_path()
        self.plot_path(path)
        
        # if there are dynamic obstacles on the planned path, replan the path
        # - update vertex 
        # - compute the shortest path
        self.fig.convas.mpl_connect('key_press_event', self.on_press)
        plt.show()

    def extract_path(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        for k in range(100):
            g_list = {}
            for x in self.get_neighbor(s):
                if not self.is_collision(s, x):
                    g_list[x] = self.g[x]
            s = min(g_list, key=g_list.get)
            path.append(s)
            if s == self.s_start:
                break

        return list(reversed(path))

    def on_press(self, event):
        """Event handler for key press event
        
        If a key is pressed, the position of the mouse is assumed to be a new obstacle. 
        - new obstacle on the path detected
        - update the node with new distance 
        - compute the shortest path
        - update the plot
        """
        x, y = event.xdata, event.ydata
        if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            print("Change position: s =", x, ",", "y =", y)

            self.visited = set()
            self.count += 1

            if (x, y) not in self.obs:
                self.obs.add((x, y))
            else:
                self.obs.remove((x, y))
                self.update_node((x, y))

            self.plot.update_obs(self.obs)

            for s_n in self.get_neighbor((x, y)):
                self.update_node(s_n)

            self.compute_shortest_path()

            plt.cla()
            self.plot.plot_grid("Lifelong Planning A*")
            self.plot_visited(self.visited)
            path = self.extract_path()
            self.plot_path(path)
            self.fig.canvas.draw_idle()
            
    def plot_path(self, path):
        px = [x[0] for x in path]
        py = [x[1] for x in path]
        plt.plot(px, py, linewidth=2)
        plt.plot(self.s_start[0], self.s_start[1], "bs")
        plt.plot(self.s_goal[0], self.s_goal[1], "gs")

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

def main():
    x_start = (5, 5)
    x_goal = (45, 25)

    lpastar = LPAStar(x_start, x_goal, "euclidean")
    lpastar.search()


if __name__ == '__main__':
    main()