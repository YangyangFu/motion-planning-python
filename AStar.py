import heapq 

from env import Map 

class AStar:
    """ A* path planning algorithm
    
    """
    def __init__(self, start, goal, heuristic_type):
        self.name = "A*"
        self.start = start
        self.goal = goal
        self.heuristic_type = heuristic_type
        
        # map related
        self.env = Map(title=self.name, type="ground_truth")
        self.env.set_start(start)
        self.env.set_goal(goal)
        self.motions = self.env.motions # feasible moving directions
        self.obstacles = self.env.obstacles # observed obstacles
        
        # initialize
        self.open = [] # priority queue
        self.visited = []
        self.parents = dict() 
        self.g_values = dict()
        
    def heuristic(self, node):
        """ Calculate the heuristic value of a node

        Args:
            node (_type_): _description_
        """
        if self.heuristic_type == "manhattan":
            return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])
        elif self.heuristic_type == "euclidean":
            return ((node[0] - self.goal[0]) ** 2 + (node[1] - self.goal[1]) ** 2) ** 0.5
    
    
    def distance(self, node1, node2):
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
    
    def f(self, node):
        """ Calculate the f value of a node

        Args:
            node (_type_): _description_
        """
        return self.g_values[node] + self.heuristic(node)
    
    def search(self):
        """ A* search algorithm
        """
        # initialize the open list with start node
        self.parents[self.start] = None
        self.g_values[self.start] = 0
        heapq.heappush(self.open, (self.f(self.start), self.start))
        
        while self.open:
            # pop
            _, current = heapq.heappop(self.open)
            
            # process current node
            self.visited.append(current)
            
            # check if the goal is reached
            if current == self.goal:
                
                return self.extract_path(current)
            
            # explore neighbors
            for motion in self.motions:
                next = (current[0] + motion[0], current[1] + motion[1])
                g_next = self.g_values[current] + self.distance(current, next)
                
                # initalize the g value of next node to infinity at the beginning
                if next not in self.g_values:
                    self.g_values[next] = float("inf")
                
                # update node
                if next not in self.visited and g_next < self.g_values[next]:
                    #self.visited.append(next)
                    self.parents[next] = current
                    self.g_values[next] = g_next
                    
                    # push to open list
                    heapq.heappush(self.open, (self.f(next), next)) 
            
        return []
    
    def extract_path(self, node):
        """ Extrack the path from start to node

        Args:
            node (_type_): _description_
        """
        path = []
        while node:
            path.append(node)
            node = self.parents[node]
            
        return path[::-1]
    
def main():
    import cv2 as cv 
    
    start = (5, 5)
    goal = (25, 45)

    astar = AStar(start, goal, "euclidean")
    astar.env.plot()
    path = astar.search()
    astar.env.plot_visited(astar.visited)
    astar.env.plot_path(path)
    cv.waitKey(0)
    
    # save a movie at the end
    astar.env.save_gif("astar.gif", 20)
    
if __name__ == '__main__':
    main()