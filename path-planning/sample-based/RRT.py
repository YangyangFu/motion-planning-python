import cv2 as cv 
import random 
import math 

from env import Map

random.seed(0)

class TreeNode():
    def __init__(self, node):
        self.val = node
        self.parent = None

class RRT():
    """Rapidly-exploring Random Tree algorithm
    """
    def __init__(self, start, goal):
        self.name = "RRT"
        self.start = TreeNode(start)
        self.goal = TreeNode(goal)
        
        # map related
        self.env = Map(title=self.name, type="ground_truth")
        self.env.set_start(start)
        self.env.set_goal(goal)
        self.motions = self.env.motions
        
        # algorithm related
        self.dt = 1 # time step
        self.max_speed = 1 # maximum motion speed on x or y axis
        self.tree = [self.start] # list of TreeNode to represent tree
        
        # visualization
        self.count = 0
        
    def search(self):
        """ generate a tree and path
        """
        
        for i in range(10000):
            # generate a random state
            state = self.random_state()
            
            # find the nearest neighbor
            nearest, min_dist = self.nearest_neighbor(state)
            
            # select an input to move from nearest to state
            input = self.select_input(nearest, state)
            
            # generate a new state
            new_state = self.new_state(nearest, input)
            
            # add new_state to tree
            if new_state and not self.has_collision(nearest.val, new_state.val):
                self.tree.append(new_state)
            
            # check if new_state is close enough to goal
            if self.distance(new_state, self.goal) < 1:
                self.goal.parent = new_state

                # extract path
                return self.extract_path()
            
            # visualization
        
        return None 
    
    def extract_path(self):
        node = self.goal 
        path = [node.val]
        
        while node.parent is not None:
            node = node.parent 
            path.append(node.val)
        
        return path[::-1]
    
    def random_state(self):
        """ Generate a random state
        """
        if random.random() > 0.05:
            return TreeNode((random.randint(0, self.env.x_lim), 
                             random.randint(0, self.env.y_lim))
                            )
        return self.goal
    
    def nearest_neighbor(self, state):
        """ Find the nearest neighbor of a state
        """
        min_dist = float("inf")
        for node in self.tree:
            dist = self.distance(node, state)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest, min_dist
    
    def distance(self, state1, state2):
        """ Calculate the distance between two states
        """
        return ((state1.val[0] - state2.val[0]) ** 2 + (state1.val[1] - state2.val[1]) ** 2) ** 0.5
    
    def select_input(self, start, to):
        """ Select an input to move from start to to
        """
        dx = to.val[0] - start.val[0]
        dy = to.val[1] - start.val[1]
        theta = math.atan2(dy, dx)
        
        # constraints on input
        ux = max(-self.max_speed, min(self.max_speed, dx/self.dt))
        uy = max(-self.max_speed, min(self.max_speed, dy/self.dt))
        
        return (ux, uy, theta)
    
    def new_state(self, state, input):
        """ Generate a new state
        """
        # input 
        ux, uy, theta = input
        
        # new state 
        x = state.val[0] + ux*self.dt
        y = state.val[1] + uy*self.dt
        
        node = TreeNode((int(x), int(y)))
        node.parent = state 
        
        return node
    
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
            
            # empty previous tree
            self.tree = [self.start]
            
            # update obstacles
            if (row, col) not in self.env.obstacles:
                self.env.obstacles.add((row, col))
                print("add obstacle at ", row, col)
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

            # plot visited 
            self.env.plot_visited([node.val for node in self.tree])
            cv.waitKey(100)
            
            # plot updated path 
            self.env.plot_path(path)
            
            self.count += 1
            print("udpate after click ", self.count)
            # show update
            cv.imshow(self.name, self.env.grid)
       
    
if __name__ == '__main__':
    start = (5, 15)
    goal = (50, 90)

    rrt = RRT(start, goal)
    rrt.plot()
    cv.waitKey(0)
     
    path = rrt.search()
    rrt.env.plot_visited([node.val for node in rrt.tree])
    cv.waitKey(0)
    rrt.env.plot_path(path)
    cv.waitKey(0)
    
    #lpastar.env.save_gif(name='lpastar.gif', duration=20)
    cv.destroyAllWindows()