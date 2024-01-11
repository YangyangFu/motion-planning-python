import cv2 as cv 
import numpy as np
import imageio 

class Map():
    
    def __init__(self, title, type='ground_truth'):
        
        self.x_lim = 31
        self.y_lim = 51
        self.motions = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]

        # type of map: ground_truth (assume know all details of the map), partial (only know from observations)
        self.type = type
        
        # for movie
        self.frames = []

        # initialize obstacles
        self.obstacles = set() # list of tuples
        self.initialize_grid_with_obstacles()

        # visualization
        self.title = title
        
        # initialize start/goal point
        self.start = (0,0)
        self.goal = (5,5)
        
    def initialize_grid_with_obstacles(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """
        x = self.x_lim
        y = self.y_lim

        if self.type == 'ground_truth':
            for i in range(y):
                self.obstacles.add((0, i))
            for i in range(y):
                self.obstacles.add((x - 1, i))

            for i in range(x):
                self.obstacles.add((i, 0))
            for i in range(x):
                self.obstacles.add((i, y-1))

            for i in range(10, 21):
                self.obstacles.add((15, i))
            for i in range(15):
                self.obstacles.add((i, 20))

            for i in range(15, 30):
                self.obstacles.add((i, 30))
            for i in range(16):
                self.obstacles.add((i, 40))

        # partial observation:
        elif self.type == 'partial':
            pass 
    
    def set_start(self, start):
        self.start = start
    
    def set_goal(self, goal):
        self.goal = goal
    
    def plot(self):
        """Initalize a new drawing window with current obstacles and start/goal points
        """
        cv.namedWindow(self.title, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.title, self.update_obstacles)
        self.draw_grid()    
        cv.imshow(self.title, self.grid)
    

    def draw_grid(self):
        # initialize a white grid 
        self.grid = np.ones((self.x_lim, self.y_lim, 3), dtype=np.uint8) * 255
        
        # add start and goal points
        self.grid[self.start[0], self.start[1]] = [255, 0, 0]
        self.grid[self.goal[0], self.goal[1]] = [0, 255, 0]
        
        # add obstacles
        for obstacle in self.obstacles:
            self.grid[obstacle[0], obstacle[1]] = [0, 0, 0]

        # save for movie
        self.frames.append(np.copy(self.grid))
        
        # show
        cv.imshow(self.title, self.grid)
        
    def update_obstacles(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            row = int(y)
            col = int(x)

            # update obstacles
            if (row, col) not in self.obstacles:
                self.obstacles.add((row, col))
            else:
                self.obstacles.remove((row, col))
                # white out the grid
                self.grid[row, col] = [255, 255, 255]
            
            # plot obstacles
            for obstacle in self.obstacles:
                self.grid[obstacle[0], obstacle[1]] = [0, 0, 0]

            # save current frame for movie
            self.frames.append(np.copy(self.grid))
            
            # replot the grid with dynamic obstacles
            self.plot()
            
            # show update
            cv.imshow(self.title, self.grid)
            
    def plot_path(self, path):
        """
        plot path on exiting cv window
        """        
        # color path with red
        for i in range(1, len(path)-2):
            cv.line(self.grid, path[i][::-1], path[i+1][::-1], (127, 127, 255), 1)
        
        self.frames.append(np.copy(self.grid))
        cv.imshow(self.title, self.grid)
        
    
    def plot_visited(self, visited):
        """
        plot visited nodes on exiting cv window
        """
        if self.start in visited:
            visited.remove(self.start)
        if self.goal in visited:
            visited.remove(self.goal)
        
        for node in visited:
            # color with gray 
            if node not in self.obstacles:
                self.grid[node[0], node[1]] = [210, 210, 210]
                
            # pause for 0.001s
            self.frames.append(np.copy(self.grid))
            cv.imshow(self.title, self.grid)
            cv.waitKey(1)
        
    
    def save_gif(self, name='grid_experiment.gif', duration=50):
        
        frames = []
        
        # resize frames by 10 times 
        for frame in self.frames:
            frames.append(cv.resize(frame, (self.y_lim*10, self.x_lim*10), interpolation=cv.INTER_AREA))
        
        print(len(frames))
        imageio.mimwrite(name, frames, duration=duration)
        
if __name__ == '__main__':
    env = Map()
    env.plot()
    cv.waitKey(0)
    cv.destroyAllWindows()
    env.save_gif()