import cv2 as cv 
import random 
import math 

from env import Map

random.seed(0)

class TreeNode():
    def __init__(self, node):
        self.val = node
        self.parent = None

class RRTStar():
    """Rapidly-exploring Random Tree Star algorithm

    """
    def __init__(self, start, goal):
        self.name = "RRTStar"
        self.start = TreeNode(start)
        self.goal = TreeNode(goal)
        
        # map related
        self.env = Map(title=self.name, type="ground_truth")
        self.env.set_start(start)
        self.env.set_goal(goal)
        self.motions = self.env.motions
        
        # algorithm related
        self.dt = 1 # time step
        self.max_speed = math.sqrt(2) # maximum motion speed on x or y axis
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
            new_state = self.steer(nearest, state, min_dist)
            
            # add new_state to tree
            if new_state and not self.has_collision(nearest.val, new_state.val):
                state_near = self.near(new_state, radius=5)
                self.tree.append(new_state)

                # connect along a mimum-cost path
                state_min = nearest
                cost_min = self.cost(state_min) + self.distance(state_min, new_state)
                for node in state_near:
                    if not self.has_collision(node.val, new_state.val):
                        cost = self.cost(node) + self.distance(node, new_state)
                        if cost < cost_min:
                            state_min = node
                            cost_min = cost
                new_state.parent = state_min

                # rewire the tree
                for node in state_near:
                    if not self.has_collision(node.val, new_state.val):
                        cost = self.cost(new_state) + self.distance(node, new_state)
                        if cost < self.cost(node):
                            node.parent = new_state
                
                
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
    
    def cost(self, state):
        """ cost of a unique path from root to state

        TODO: the implementation is not efficient and can be improved by maintaining a cost attribute in TreeNode
        """
        cost = 0
        while state.parent is not None:
            cost += self.distance(state, state.parent)
            state = state.parent
        return cost

    def steer(self, nearest, state, min_dist):
        """ Steer from nearest to state based on motion constraints
        """
        if min_dist < self.max_speed:
            return state
        else:
            dx = state.val[0] - nearest.val[0]
            dy = state.val[1] - nearest.val[1]
            theta = math.atan2(dy, dx)
            
            x = nearest.val[0] + self.max_speed*math.cos(theta)
            y = nearest.val[1] + self.max_speed*math.sin(theta)
            
            new_state = TreeNode((int(x), int(y)))
            new_state.parent = nearest
            return new_state

    def near(self, state, radius=1):
        """ finds the vertices in the tree that are within a ball of radius centered at state
        """
        near = []
        for node in self.tree:
            if self.distance(node, state) <= radius:
                near.append(node)
        return near

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

    rrt = RRTStar(start, goal)
    rrt.plot()
    cv.waitKey(0)
     
    path = rrt.search()
    rrt.env.plot_visited([node.val for node in rrt.tree])
    cv.waitKey(0)
    rrt.env.plot_path(path)
    cv.waitKey(0)
    
    #lpastar.env.save_gif(name='lpastar.gif', duration=20)
    cv.destroyAllWindows()