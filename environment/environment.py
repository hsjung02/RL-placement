import numpy as np
import torch as th
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


class CircuitEnv(Env):
    def __init__(self,
                 adjacency_matrix: List,
                 cells: Dict,
                 macro_indices: List,
                 std_indices: List,
                 pin_indices: Optional[List],
                 canvas_size: int = 32,
                 reward_weights: List =[1,0,0]) -> None:
        super().__init__()

        # Properties of netlist and canvas
        self.init_properties(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, canvas_size, reward_weights)
        # Properties about placement
        self.init_placement()
        # Properties about current canvas state
        self.init_canvas()

        # Static features: features that do not change over time
        self.static_features = self.get_static_features()

        edge_num = np.sum(self.adjacency_matrix)

        self.action_space = spaces.Discrete(canvas_size**2)
        self.observation_space = spaces.Dict({
            "metadata":spaces.Box(low=0, high=1, shape=(10,), dtype=np.float64),
            "nodes":spaces.Box(low=-1, high=100, shape=(len(self.cells),8), dtype=np.float64),
            "adj_i":spaces.Box(low=0, high=4, shape=(edge_num,), dtype=np.int32),
            "adj_j":spaces.Box(low=0, high=4, shape=(edge_num,), dtype=np.int32),
            "current_node":spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)
        })
    
    def reset(self):
        self.init_placement()
        self.init_canvas()
        return self.get_all_features(), {} # Return empty dictionary just to follow gym Env API
        
    def step(self, action: int):

        # Place the macro to the 'action' spot
        self.place_macro(action)
        
        features = self.get_all_features()

        # Next macro
        self.macro_idx_in_macro_array += 1


        # Macro lacement done
        if self.macro_idx_in_macro_array == self.macro_num:
            self.done = True
            # self.place_std()

        # Set reward: weighted sum of wirelength, congestion if done, 0 otherwise
        if not self.done:
            reward = 0
        else:
            self.place_std()
            reward = self.get_reward()

        info = {}
        
        return features, reward, self.done, False, info


    def render(self):
        self.show_canvas()
        return

    def get_static_features(self) -> Dict:
        # Static features
        features = {}

        # Metadata contains 10 features:
        # number of edges, number of hard macros, number of soft macros, number of port clusters,
        # horizontal routes per micron, vertical routes per micron,
        # macro vertical routing allocation,
        # grid columns, grid rows
        # Here we just ignore
        # TODO: Add metadata
        metadata = []
        features["metadata"] = np.zeros(10)
        #features["metadata"] = th.tensor(metadata)


        # Static featues of every nodes
        # width, height, is hard macro, is soft macro, is port cluster
        node_static = [[self.cells[macro]['width'], self.cells[macro]['height'], 1, 0, 0] for macro in self.macro_indices] + [[self.cells[std]['width'], self.cells[std]['height'], 0, 1, 0] for std in self.std_indices] + [[0, 0, 0, 1, 0] for pin in self.pin_indices]
        features["node_static"] = node_static

        # Express edges using adj_i and adj_j array
        # if 3 edges connecting (0,1), (0,2), (1,2),
        # then adj_i = [0,0,1,1,2,2], adj_j = [1,2,2,0,0,1]
        # using adj_i and adj_j increases data efficiency
        # and simplify operations
        adj_i = []
        adj_j = []
        for i in range(len(self.adjacency_matrix)):
            for j in range(len(self.adjacency_matrix)):
                if self.adjacency_matrix[i][j] != 0:
                    adj_i.append(i)
                    adj_j.append(j)
        features["adj_i"] = np.array(adj_i)
        features["adj_j"] = np.array(adj_j)

        return features

    def get_dynamic_features(self) -> Dict:
        # Dynamic features that changes as placement precedes
        features = {}

        # Dynamics features of every nodes
        # x position, y position, is placed
        node_dynamic = [[self.cell_position[cell][0], self.cell_position[cell][1], 1 if cell<self.macro_idx_in_macro_array else 0] for cell in self.cells]
        features["node_dynamic"] = node_dynamic

        # Current node to be placed
        features["current_node"] = np.array([self.macro_idx_in_macro_array])

        return features

    def get_all_features(self):
        # Merge static features and dynamic features

        static_features = self.static_features
        dynamic_features = self.get_dynamic_features()
        features = {}
        features["metadata"] = static_features["metadata"]
        features["adj_i"] = static_features["adj_i"]
        features["adj_j"] = static_features["adj_j"]
        features["current_node"] = dynamic_features["current_node"]

        node_static = static_features["node_static"]
        node_dynamic = dynamic_features["node_dynamic"]

        nodes = [node_static[i]+node_dynamic[i] for i in range(len(self.cells))]
        features["nodes"] = np.array(nodes)

        return features
    
    def place_macro(self, action: int) -> None:

        # Node index of current node. Usually node_idx==self.macro_idx_in_macro_array
        node_idx = self.macro_indices[self.macro_idx_in_macro_array]
        # Convert action to (x,y) coordinate
        action = [action//self.canvas_size, action%self.canvas_size]

        # Fill canvas and density grid, handle edges cases
        for y in range(action[0]-int(self.cells[node_idx]['height']-1)//2, action[0]+int(self.cells[node_idx]['height'])//2+1):
            for x in range(action[1]-int(self.cells[node_idx]['width']-1)//2,action[1]+int(self.cells[node_idx]['width'])//2+1):
                try:
                    self.canvas[y][x] = node_idx
                except IndexError:
                    pass
                try:
                    if y>0 and x>0:
                        self.density_grid[y-1][x-1] += 1
                except IndexError:
                    pass
                try:
                    if y>0:
                        self.density_grid[y-1][x] += 1
                except IndexError:
                    pass
                try:
                    if x>0:
                        self.density_grid[y][x-1] += 1
                except IndexError:
                    pass
                try:
                    self.density_grid[y][x] += 1
                except IndexError:
                    pass

        # Redundant
        self.canvas[action[0]][action[1]] = node_idx
        # Save cell position
        self.cell_position[node_idx] = action

        self.action_history.append(action)

    
    def action_masks(self):
        # Consider invalid actions during training and inference
        node_idx = self.macro_idx_in_macro_array
        mask = np.array([[1 if i==-1 else 0 for i in j] for j in self.canvas])
        

        # Cannot place near edges
        mask[0:int(self.cells[node_idx]['height']-1)//2,0:self.canvas_size] = 0
        mask[self.canvas_size-1-int(self.cells[node_idx]['height']-1)//2: self.canvas_size,0:self.canvas_size] = 0
        mask[0:self.canvas_size,0:int(self.cells[node_idx]['width']-1)//2] = 0
        mask[0:self.canvas_size,self.canvas_size-1-int(self.cells[node_idx]['width']-1)//2: self.canvas_size] = 0

        # Density constraints
        dense = np.where(self.density_grid>=2)
        s = len(dense[0])
        for x in range(s):
            i = dense[0][x]
            j = dense[1][x]
            mask[max(0, i-int(self.cells[node_idx]['height'])//2-1):min(self.canvas_size,i+int(self.cells[node_idx]['height']+1)//2+1), max(0, j-int(self.cells[node_idx]['width'])//2-1):min(self.canvas_size,j+int(self.cells[node_idx]['width']+1)//2+1)] = 0

        return mask.reshape(1,-1)
        
    def place_std(self) -> None:
        # TODO: Implement ePlAce algorithm
        self.eplace()

    def eplace(self) -> None:

        canvas_x = 2716400/2000
        canvas_y = 2650880/2000


        hard_macro_num = len(self.macro_indices)
        soft_macro_num = len(self.std_indices)
        pin_num = len(self.pin_indices)

        # Initial random placement
        soft_macro_position_x = (canvas_x)*np.random.rand(soft_macro_num)
        soft_macro_position_y = (canvas_y)*np.random.rand(soft_macro_num)

        # Hard macro position and pin positions are fixed
        hard_macro_position_x = np.array([self.cell_position[c][1]*self.grid_width for c in self.macro_indices])
        hard_macro_position_y = np.array([self.cell_position[c][0]*self.grid_height for c in self.macro_indices])
        pin_position_x = np.array([self.cells[c]['x']*self.grid_width for c in self.pin_indices])
        pin_position_y = np.array([self.cells[c]['y']*self.grid_height for c in self.pin_indices])

        # Concatenate hard macro, soft macro, pin positions
        cell_position_x = np.hstack([hard_macro_position_x, soft_macro_position_x, pin_position_x])
        cell_position_y = np.hstack([hard_macro_position_y, soft_macro_position_y, pin_position_y])


        cell_charge = np.zeros(hard_macro_num+soft_macro_num+pin_num)
        for i in range(hard_macro_num):
            cell_charge[i] = self.cells[i]['width']*self.cells[i]['height']
        cell_charge[hard_macro_num:] = self.cells[hard_macro_num+1]['width']*self.cells[hard_macro_num+1]['height']

        max_iteration = 1
        dt = 1
        mass = 1

        # Initial state
        self.std_position_x = cell_position_x[hard_macro_num:hard_macro_num+soft_macro_num]
        self.std_position_y = cell_position_y[hard_macro_num:hard_macro_num+soft_macro_num]
        self.show_canvas()

        for _ in range(max_iteration):

            # Evaluate force_attractive_x, force_attractive_y
            force_attractive_x = np.sum(np.abs(self.adjacency_matrix*cell_position_x - cell_position_x.reshape(hard_macro_num+soft_macro_num+pin_num,1)), axis=1, where=self.adjacency_matrix!=0)
            force_attractive_y = np.sum(np.abs(self.adjacency_matrix*cell_position_y - cell_position_y.reshape(hard_macro_num+soft_macro_num+pin_num,1)), axis=1, where=self.adjacency_matrix!=0)
            
            # Evaluate force_repulsive_x, force_repulsive_y
            cell_grid_x = (cell_position_x/self.grid_width).astype(int) - 1
            cell_grid_y = (cell_position_y/self.grid_height).astype(int) - 1
            cell_grid = np.vstack([cell_grid_y, cell_grid_x])
            # Charge for each grid
            grid_charge = np.zeros((self.canvas_size, self.canvas_size))
            np.add.at(grid_charge, tuple(cell_grid), cell_charge)

            # Generate dx, dy, r_sq array for iteration
            x, y = np.meshgrid(range(self.canvas_size), range(self.canvas_size))

            x = x.flatten()
            y = y.flatten()
            q = grid_charge.flatten()
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            q = q.reshape(-1, 1)

            dx = x - x.T
            dy = y - y.T
            r_sq = dx**2 + dy**2
            np.fill_diagonal(r_sq, 1e8)
            denom = np.sqrt(r_sq)**3

            # Apply Coulomb's law
            fx = np.sum(q*dx/denom, axis=1)
            fy = np.sum(q*dy/denom, axis=1)

            force_repulsive_x = fx[cell_grid_y*self.canvas_size+cell_grid_x]
            force_repulsive_y = fy[cell_grid_y*self.canvas_size+cell_grid_y]

            # Calculate net force using attractive force(wirelength) and repulsive force
            force_x = - force_attractive_x + force_repulsive_x
            force_y = - force_attractive_y + force_repulsive_y
            # Get acceleration for std cells
            ax = force_x[hard_macro_num:hard_macro_num+soft_macro_num]/mass
            ay = force_y[hard_macro_num:hard_macro_num+soft_macro_num]/mass
            # Move cells using acceleration
            cell_position_x[hard_macro_num:hard_macro_num+soft_macro_num] += 0.5*ax*(dt**2)
            cell_position_y[hard_macro_num:hard_macro_num+soft_macro_num] += 0.5*ay*(dt**2)
            # Avoid getting out of the canvas by clipping positions
            cell_position_x = np.clip(cell_position_x, a_min=0, a_max=self.grid_width*self.canvas_size)
            cell_position_y = np.clip(cell_position_y, a_min=0, a_max=self.grid_height*self.canvas_size)
        
        # Save eplace result
        self.std_position_x = cell_position_x[hard_macro_num:hard_macro_num+soft_macro_num]
        self.std_position_y = cell_position_y[hard_macro_num:hard_macro_num+soft_macro_num]
        for i in range(hard_macro_num, hard_macro_num+soft_macro_num):
            self.cell_position[i] = [cell_position_y[i], cell_position_x[i]]

    def get_reward(self) -> int:
        # Weighted sum of wirelength, congestion and density
        # density = 0
        cost = np.array([self.get_wirelength(), self.get_congestion(), self.get_density()], dtype=np.float32)
        return - cost @ self.reward_weights
        return - cost @ self.reward_weights - self.edge_penalty() # edge penalty added to place in the center

    def get_wirelength(self) -> int:
        # Get HPWL from two connected cells and sum up them
        wirelength = 0
        for i in range(len(self.adjacency_matrix)):
            if self.cell_position[i][0] == -1 and self.cell_position[i][1] == -1:
                continue
            for j in range(i,len(self.adjacency_matrix[0])):
                if self.cell_position[j][0] == -1 and self.cell_position[j][1] == -1:
                    continue
                if self.adjacency_matrix[i,j] >0 and i != j:
                    wirelength += abs(self.cell_position[i][0]-self.cell_position[j][0])+abs(self.cell_position[i][1]-self.cell_position[j][1])+2
        return wirelength

    def get_congestion(self) -> int:
        return 0
        # Route following right-angle algorithm
        routing_grid = np.array([[0 for i in range(self.canvas_size-1)] for j in range(self.canvas_size-1)])

        for cell1 in range(self.cell_num):
            connected = np.where(np.array(self.adjacency_matrix[cell1])!=0)[0]

            for cell2 in connected:
                if cell2 <= cell1:
                    continue
                routing_type = np.random.randint(2)
                position_type = (self.cell_position[cell1][1]<=self.cell_position[cell2][1]) + (self.cell_position[cell1][0]<=self.cell_position[cell2][0])
                x1 = min(self.cell_position[cell1][1], self.cell_position[cell2][1])
                x2 = max(self.cell_position[cell1][1], self.cell_position[cell2][1])
                y1 = min(self.cell_position[cell1][0], self.cell_position[cell2][0])
                y2 = max(self.cell_position[cell1][0], self.cell_position[cell2][0])
                if 31 in [x1, x2, y1, y2]:
                    continue
                if routing_type%2 == 0:
                    routing_grid[y1, x1:x2] += 1
                    if (routing_type+position_type)%2==0:
                        routing_grid[y1:y2, x2] += 1
                    else:
                        routing_grid[y1:y2, x1] += 1
                else:
                    routing_grid[y2, x1:x2] += 1
                if (routing_type+position_type)%2==0:
                    routing_grid[y1:y2, x2] += 1
                else:
                    routing_grid[y1:y2, x1] += 1

        # Congestion defined as maximum value in the routing grid
        congestion = np.max(routing_grid)

        return congestion

    def get_density(self) -> int:
        # arbitrary set to 0
        return 0

    def show_canvas(self) -> None:
        # Render function
        image = np.array([[self.color_list[self.canvas[j//8][i//8]+1] for i in range(8*self.canvas_size)] for j in range(8*self.canvas_size)])
        # for i in range(self.cell_num):
        #     for j in range(i+1, self.cell_num):
        #         if self.adjacency_matrix[i,j] != 0:
        #             y = [8*self.cell_position[i][0]+3, 8*self.cell_position[j][0]+3]
        #             x = [8*self.cell_position[i][1]+3, 8*self.cell_position[j][1]+3]
        #             plt.plot(x, y, color="red", linewidth=0.8, alpha=0.7)
        plt.scatter(8*self.std_position_x/self.grid_width, 8*self.std_position_y/self.grid_height, s=0.1)
        # if mode=="show":
        #   plt.text(50,270,"HPWL: "+str(self.get_wirelength()), size="xx-large")
        #   plt.text(50,285,"Congestion: "+str(self.get_congestion()), size="xx-large")
        #   plt.text(50,300,"Reward: "+str(self.get_reward()), size="xx-large")
        plt.text(50,270,"HPWL: "+str(self.get_wirelength()), size="xx-large")
        plt.text(50,285,"Congestion: "+str(self.get_congestion()), size="xx-large")
        plt.text(50,300,"Reward: "+str(self.get_reward()), size="xx-large")
        for i in range(self.canvas_size):
            plt.plot([0,8*32], [8*i-1,8*i-1],c = 'gray', linestyle = '--', linewidth=0.5)
        for j in range(self.canvas_size):
            plt.plot([8*j-1,8*j-1], [0,8*32], c = 'gray', linestyle = '--', linewidth=0.5)
        fig = plt.imshow(image)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # if mode=="save":
        #     plt.savefig(path)
        return

    def init_properties(self,
                        adjacency_matrix: List,
                        cells: Dict,
                        macro_indices: List,
                        std_indices: List,
                        pin_indices: Optional[List],
                        canvas_size: int = 32,
                        reward_weights: List = [1,0,0]) -> None:
        
        # Adjacency matrix with all weights 1
        self.adjacency_matrix = np.array(adjacency_matrix)
        # Width and height information of all cells
        # {0:{'width':8,'height':8}, 1:{'width':8,'height':8}}
        self.cells = cells
        # List of macro cell indices
        self.macro_indices = macro_indices
        # List of std cell indices
        self.std_indices = std_indices
        # List of pins(=ports) indices
        self.pin_indices = pin_indices
        # Canvas size set to 32
        self.canvas_size = canvas_size
        # Reward function weights
        self.reward_weights = reward_weights

        self.cell_num = len(cells)

        # Cell color when rendering
        self.color_list = [np.random.randint(256, size=3) for i in range(self.cell_num+2)]
        self.color_list[0] = [255,255,255]
    
    def init_placement(self) -> None:
        # Current macro index to place
        self.macro_idx_in_macro_array = 0
        # Number of hard macro cells
        self.macro_num = len(self.macro_indices)
        # Placement done
        self.done = False
        # Action history
        self.action_history = []

    def init_canvas(self) -> None:
        # Value in canvas 2D array represents which cell is placed there
        self.canvas = np.array([[-1 for i in range(self.canvas_size)] for j in range(self.canvas_size)])
        macro_position = {macro:[-1,-1] for macro in self.macro_indices}
        std_position = {std:[-1,-1] for std in self.std_indices}
        pin_position = {pin:[self.cells[pin]['y'], self.cells[pin]['x']] for pin in self.pin_indices}
        # Position of cells
        self.cell_position = macro_position | std_position | pin_position
        # Density grid for density constraint and preventing overlaps
        self.density_grid = np.array([[0 for i in range(self.canvas_size-1)] for j in range(self.canvas_size-1)])
        self.std_position_x = np.array([])
        self.std_position_y = np.array([])
        self.grid_width = 2716400/2000/self.canvas_size
        self.grid_height = 2650880/2000/self.canvas_size