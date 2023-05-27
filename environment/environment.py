import numpy as np
import torch as th
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import random

class CircuitEnv(Env):
    def __init__(self,
                 adjacency_matrix: List,
                 cells: Dict,
                 macro_indices: List,
                 std_indices: List,
                 pin_indices: Optional[List],
                 canvas_size: int = 32,
                 reward_weights: List =[1,0]) -> None:
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
            "adj_i":spaces.Box(low=0, high=7000, shape=(edge_num,), dtype=np.int32),
            "adj_j":spaces.Box(low=0, high=7000, shape=(edge_num,), dtype=np.int32),
            "current_node":spaces.Box(low=0, high=16, shape=(1,), dtype=np.int32)
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

        # Set reward: weighted sum of wirelength, congestion if done, 0 otherwise
        if not self.done:
            reward = 0
        else:
            self.place_std()
            reward = self.get_reward()

        info = {}
        
        return features, reward, self.done, False, info

    def render(self, mode="show", path=""):
        # Render function
        # fig, ax = plt.subplots(1)
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
        plt.text(50,270,"HPWL: "+str(int(self.get_wirelength())), size="xx-large")
        plt.text(50,285,"Congestion: "+str(self.get_congestion()), size="xx-large")
        plt.text(50,300,"Reward: "+str(int(self.get_reward())), size="xx-large")
        for i in range(self.canvas_size):
            plt.plot([0,8*32], [8*i-1,8*i-1],c = 'gray', linestyle = '--', linewidth=0.5)
        for j in range(self.canvas_size):
            plt.plot([8*j-1,8*j-1], [0,8*32], c = 'gray', linestyle = '--', linewidth=0.5)
        fig = plt.imshow(image)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # plt.show()
        #if mode=="save":
        #plt.savefig(path)
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        # ax.imshow(image)
        # ax.axis('off')
        if mode=="show":
            plt.show()
        elif mode=="rgb_array":
            canvas = FigureCanvas(fig)
            canvas.draw()
            
            image_from_plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)
            print(type(image_from_plot))
            return image_from_plot
        elif mode=="save":
            plt.savefig(path)

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
        # adj_i = []
        # adj_j = []
        # for i in range(len(self.adjacency_matrix)):
        #     for j in range(len(self.adjacency_matrix)):
        #         if self.adjacency_matrix[i][j] != 0:
        #             adj_i.append(i)
        #             adj_j.append(j)
        # features["adj_i"] = np.array(adj_i)
        # features["adj_j"] = np.array(adj_j)
        adj_i, adj_j = np.nonzero(self.adjacency_matrix)
        features["adj_i"] = adj_i
        features["adj_j"] = adj_j

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
        for y in range(action[0]-int(self.cells[node_idx]['height']//2), action[0]+int((self.cells[node_idx]['height']+1)//2)+1):
            for x in range(action[1]-int(self.cells[node_idx]['width']//2), action[1]+int((self.cells[node_idx]['width']+1)//2)+1):
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
        mask[0:int(self.cells[node_idx]['height']//2), :] = 0
        mask[self.canvas_size-int((self.cells[node_idx]['height']+1)//2):self.canvas_size, :] = 0
        mask[:, 0:int(self.cells[node_idx]['width']//2)] = 0
        mask[:, self.canvas_size-int((self.cells[node_idx]['width']+1)//2):self.canvas_size] = 0

        # Density constraints
        dense = np.where(self.density_grid>=2)
        s = len(dense[0])
        for x in range(s):
            i = dense[0][x]
            j = dense[1][x]
            mask[max(0, i-int((self.cells[node_idx]['height']+1)//2)):min(self.canvas_size,i+int((self.cells[node_idx]['height'])//2)+2),
                 max(0, j-int((self.cells[node_idx]['width']+1)//2)):min(self.canvas_size,j+int((self.cells[node_idx]['width'])//2)+2)] = 0

        return mask.reshape(1,-1)
        
    def place_std(self) -> None:
        self.eplace()

    def eplace(self) -> None:
        canvas_x = 2716400/2000
        canvas_y = 2650880/2000

        
        
        #parameters===================================================
        #합이 1이 되도록 맞출것
        weight_attractive=0.2
        weight_repulsive=1-weight_attractive

        max_iteration = 20
        dt = 0.5
        mass = 1
        
        #scale parameters
        #force_scale = 1e+4 #for random force
        repulsive_force_scale = 1e+7
        position_scale = 1e-6
        ePlace_grid_force_scale = 200
        hook_constant = 1e+4
        #=============================================================


        hard_macro_num = len(self.macro_indices)
        soft_macro_num = len(self.std_indices)
        pin_num = len(self.pin_indices)

        # Initial random placement 
        soft_macro_position_x = np.random.rand(soft_macro_num)*self.grid_width *(self.canvas_size)
        soft_macro_position_y = np.random.rand(soft_macro_num)*self.grid_height*(self.canvas_size)
        # soft_macro_position_x = th.rand(soft_macro_num)*self.grid_width *(self.canvas_size)
        # soft_macro_position_y = th.rand(soft_macro_num)*self.grid_height*(self.canvas_size)
        # Hard macro position and pin positions are fixed
        hard_macro_position_x = np.array([self.cell_position[c][1]*self.grid_width for c in self.macro_indices])
        hard_macro_position_y = np.array([self.cell_position[c][0]*self.grid_height for c in self.macro_indices])
        # hard_macro_position_x = th.tensor([self.cell_position[c][1]*self.grid_width for c in self.macro_indices])
        # hard_macro_position_y = th.tensor([self.cell_position[c][0]*self.grid_height for c in self.macro_indices])
        pin_position_x = np.array([self.cells[c]['x']*self.grid_width for c in self.pin_indices])
        pin_position_y = np.array([self.cells[c]['y']*self.grid_height for c in self.pin_indices])
        # pin_position_x = th.tensor([self.cells[c]['x']*self.grid_width for c in self.pin_indices])
        # pin_position_y = th.tensor([self.cells[c]['y']*self.grid_height for c in self.pin_indices])

        # Concatenate hard macro, soft macro, pin positions
        cell_position_x = np.hstack([hard_macro_position_x, soft_macro_position_x, pin_position_x])
        cell_position_y = np.hstack([hard_macro_position_y, soft_macro_position_y, pin_position_y])
        # cell_position_x = th.hstack([hard_macro_position_x, soft_macro_position_x, pin_position_x])
        # cell_position_y = th.hstack([hard_macro_position_y, soft_macro_position_y, pin_position_y])
        for i in range(hard_macro_num, hard_macro_num+soft_macro_num):
                self.cell_position[i] = [cell_position_y[i], cell_position_x[i]]
        
        cell_grid_position_x = np.array(hard_macro_num + soft_macro_num + pin_num)
        cell_grid_position_y = np.array(hard_macro_num + soft_macro_num + pin_num)
        # cell_grid_position_x = th.array(hard_macro_num + soft_macro_num + pin_num)
        # cell_grid_position_y = th.array(hard_macro_num + soft_macro_num + pin_num)
        
        soft_macro_position_delta_x = np.zeros(soft_macro_num)
        soft_macro_position_delta_y = np.zeros(soft_macro_num)
        # soft_macro_position_delta_x = th.zeros((1,soft_macro_num))
        # soft_macro_position_delta_y = th.zeros((1,soft_macro_num))

        cell_charge = np.zeros(hard_macro_num+soft_macro_num+pin_num)
        # cell_charge = th.zeros((1,hard_macro_num+soft_macro_num+pin_num))
        for i in range(hard_macro_num):
            cell_charge[i] = self.cells[i]['width']*self.cells[i]['height']
        cell_charge[hard_macro_num:] = self.cells[hard_macro_num+1]['width']*self.cells[hard_macro_num+1]['height']

        
        ePlace_grid_force_x = np.zeros((32,32)) #해당 그리드에 셀이 위치할 경우 받는 x방향 힘
        ePlace_grid_force_y = np.zeros((32,32)) #해당 그리드에 셀이 위치할 경우 받는 y방향 힘
        # ePlace_grid_force_x = th.zeros((32,32)) #해당 그리드에 셀이 위치할 경우 받는 x방향 힘
        # ePlace_grid_force_y = th.zeros((32,32)) #해당 그리드에 셀이 위치할 경우 받는 y방향 힘
        
        #calculate grid force==================================================
        canvas = np.array(self.canvas)
        ePlace_grid_force_x = np.zeros(canvas.shape)
        ePlace_grid_force_y = np.zeros(canvas.shape)
        # canvas = th.tensor(self.canvas)
        # ePlace_grid_force_x = th.zeros((1,canvas.shape))
        # ePlace_grid_force_y = th.zeros((1,canvas.shape))

        # Create masks for each condition
        mask_canvas = canvas[1:30, 1:30] != -1

        mask_x_greater_right = np.roll(canvas, -1, axis=1)[1:30, 1:30] < canvas[1:30, 1:30]
        mask_x_greater_left = np.roll(canvas, 1, axis=1)[1:30, 1:30] < canvas[1:30, 1:30]

        mask_y_greater_down = np.roll(canvas, -1, axis=0)[1:30, 1:30] < canvas[1:30, 1:30]
        mask_y_greater_up = np.roll(canvas, 1, axis=0)[1:30, 1:30] < canvas[1:30, 1:30]

        # Apply conditions
        ePlace_grid_force_x[1:30, 1:30] = np.where(mask_x_greater_right & mask_canvas, 1, 
                                                np.where(mask_x_greater_left & mask_canvas, -1, 
                                                            np.where(mask_canvas, 0.5, 0)))

        ePlace_grid_force_y[1:30, 1:30] = np.where(mask_y_greater_down & mask_canvas, 1, 
                                                np.where(mask_y_greater_up & mask_canvas, -1, 0))

       
        boundary_grid_force = 0.2
        ePlace_grid_force_x[:,0]  +=  boundary_grid_force
        ePlace_grid_force_x[:,31] += -boundary_grid_force
        ePlace_grid_force_y[0,:]  +=  boundary_grid_force
        ePlace_grid_force_y[31,:] += -boundary_grid_force
        #======================================================================
        
        
        # Initial state
        # self.std_position_x = cell_position_x[hard_macro_num:hard_macro_num+soft_macro_num]
        # self.std_position_y = cell_position_y[hard_macro_num:hard_macro_num+soft_macro_num]
        # self.render()


        grid_y, grid_x = np.mgrid[0:32, 0:32]



        for iter in range(1, max_iteration+1):

            #마지막 3회의 iteration에서는 overlap 방지를 위해 척력만 고려
            if (iter>max_iteration-3):
                weight_attractive = 0
                weight_repulsive = 1
            
            #모든 셀의 위치 차이
            cell_position_x_diff = np.subtract.outer(cell_position_x, cell_position_x) #delta x 배열
            cell_position_y_diff = np.subtract.outer(cell_position_y, cell_position_y) #delta y 배열
            cell_position_distance = np.power(np.square(cell_position_x_diff)+np.square(cell_position_y_diff), 3/2) #r^1.5 배열
            cell_position_distance[cell_position_distance==0] = np.inf

            #인력 계산=======================================================================================================
            
            elementwise_attractive_force_x = np.multiply(hook_constant * cell_position_x_diff, self.adjacency_matrix)
            elementwise_attractive_force_y = np.multiply(hook_constant * cell_position_y_diff, self.adjacency_matrix)
            force_attractive_x = np.sum(elementwise_attractive_force_x, axis=1)
            force_attractive_y = np.sum(elementwise_attractive_force_y, axis=1)
            #================================================================================================================

            #척력 계산=======================================================================================================
            
            elementwise_repulsive_force_x = np.divide(cell_position_x_diff, cell_position_distance) #delta_x/r^3
            force_repulsive_x = np.sum(elementwise_repulsive_force_x, axis=0) #척력은 column 원소들끼리 더한다
            force_repulsive_x = np.multiply(force_repulsive_x, cell_charge) 

            elementwise_repulsive_force_y = np.divide(cell_position_y_diff, cell_position_distance)
            force_repulsive_y = np.sum(elementwise_repulsive_force_y, axis=0) #척력은 column 원소들끼리 더한다
            force_repulsive_y = np.multiply(force_repulsive_y, cell_charge)
                                                            
            #================================================================================================================

            #그리드로 인해 받는 힘=============================================================================================
            
            #그리드 위치
            cell_grid_position_x = cell_position_x / self.grid_width
            cell_grid_position_x = np.asarray(cell_grid_position_x, dtype=int)
            cell_grid_position_y = cell_position_y / self.grid_height
            cell_grid_position_y = np.asarray(cell_grid_position_y, dtype=int)
            #그리드
            current_grid_force_x = ePlace_grid_force_x * ePlace_grid_force_scale
            current_grid_force_y = ePlace_grid_force_y * ePlace_grid_force_scale
            # Create mask arrays
            mask_x = cell_grid_position_x[:, None, None] == grid_x
            mask_y = cell_grid_position_y[:, None, None] == grid_y
            # Combine masks
            mask = mask_x & mask_y
            # Update force_repulsive_x and force_repulsive_y
            force_repulsive_x -= np.sum(mask * current_grid_force_x, axis=(1, 2))
            force_repulsive_y -= np.sum(mask * current_grid_force_y, axis=(1, 2))
                
            #================================================================================================================

            #repulsive_force scale 조정======================================================================================
            force_repulsive_x = force_repulsive_x * repulsive_force_scale
            force_repulsive_y = force_repulsive_y * repulsive_force_scale
            #================================================================================================================
           
            # Calculate net force using attractive force(wirelength) and repulsive force
            force_x = -weight_attractive * force_attractive_x - weight_repulsive * force_repulsive_x
            force_y = -weight_attractive * force_attractive_y - weight_repulsive * force_repulsive_y
            # Get acceleration for std cells
            ax = force_x[hard_macro_num:hard_macro_num+soft_macro_num]
            ay = force_y[hard_macro_num:hard_macro_num+soft_macro_num]
            
            
            # Move cells using acceleration
            soft_macro_position_delta_x = (0.5*(dt**2)/mass*position_scale)*ax
            soft_macro_position_delta_y = (0.5*(dt**2)/mass*position_scale)*ay
            # soft_macro_position_delta_x = (0.5*ax*(dt**2))*position_scale
            # soft_macro_position_delta_y = (0.5*ay*(dt**2))*position_scale


            cell_position_x[hard_macro_num : hard_macro_num + soft_macro_num] += soft_macro_position_delta_x
            cell_position_y[hard_macro_num : hard_macro_num + soft_macro_num] += soft_macro_position_delta_y
            
            # cell_position_x[hard_macro_num : hard_macro_num + soft_macro_num] = np.add(cell_position_x[hard_macro_num : hard_macro_num + soft_macro_num], soft_macro_position_delta_x)
            # cell_position_y[hard_macro_num : hard_macro_num + soft_macro_num] = np.add(cell_position_y[hard_macro_num : hard_macro_num + soft_macro_num], soft_macro_position_delta_y)

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
        cost = np.array([self.get_wirelength(), self.get_congestion()], dtype=np.float32)
        return - cost @ self.reward_weights

    def get_wirelength(self) -> int:
        # Get HPWL from two connected cells and sum up them
        cell_positions = np.array(self.cell_position)
        expanded_positions = cell_positions[:, np.newaxis, :]
        diffs = np.abs(expanded_positions - cell_positions)
        distances = np.sum(diffs, axis=-1)
        wirelength = np.sum(distances*self.adjacency_matrix)

        return wirelength

    def get_congestion(self) -> int:
        # Route following right-angle algorithm
        vertical_routing_grid = np.zeros((self.canvas_size-1, self.canvas_size))
        horizontal_routing_grid = np.zeros((self.canvas_size, self.canvas_size-1))

        adj_i = self.static_features["adj_i"]
        adj_j = self.static_features["adj_j"]
        n = len(adj_i)

        for i in range(n):
            cell1 = adj_i[i]
            cell2 = adj_j[i]
            if cell2 <= cell1:
                continue
            x1 = int(min(self.cell_position[cell1][1], self.cell_position[cell2][1])/self.grid_width-1/self.grid_width)
            x2 = int(max(self.cell_position[cell1][1], self.cell_position[cell2][1])/self.grid_width-1/self.grid_width)
            y1 = int(min(self.cell_position[cell1][0], self.cell_position[cell2][0])/self.grid_height-1/self.grid_height)
            y2 = int(max(self.cell_position[cell1][0], self.cell_position[cell2][0])/self.grid_height-1/self.grid_height)
            routing_type = np.random.randint(2)
            position_type = (self.cell_position[cell1][1]<=self.cell_position[cell2][1]) + (self.cell_position[cell1][0]<=self.cell_position[cell2][0])

            if routing_type%2 == 0:
                horizontal_routing_grid[y1, x1:x2] += 1
            else:
                horizontal_routing_grid[y2, x1:x2] += 1
            if (routing_type+position_type)%2==0:
                vertical_routing_grid[y1:y2, x2] += 1
            else:
                vertical_routing_grid[y1:y2, x1] += 1

        # Congestion defined as maximum value in the routing grid
        congestion = np.max(vertical_routing_grid) + np.max(horizontal_routing_grid)

        return congestion

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
        # macro_position = {macro:[-1,-1] for macro in self.macro_indices}
        # std_position = {std:[-1,-1] for std in self.std_indices}
        # pin_position = {pin:[self.cells[pin]['y'], self.cells[pin]['x']] for pin in self.pin_indices}
        # # Position of cells
        # #self.cell_position = macro_position | std_position | pin_position
        

        # temp={}
        # temp.update(macro_position)
        # temp.update(std_position)
        # temp.update(pin_position)
        # self.cell_position = temp
        self.cell_position = [[-1,-1] for _ in range(len(self.macro_indices)+len(self.std_indices))] + [[self.cells[pin]['y'], self.cells[pin]['x']] for pin in self.pin_indices]

        
        # Density grid for density constraint and preventing overlaps
        self.density_grid = np.array([[0 for i in range(self.canvas_size-1)] for j in range(self.canvas_size-1)])
        self.std_position_x = np.array([])
        self.std_position_y = np.array([])
        self.grid_width = 2716400/1000/self.canvas_size
        self.grid_height = 2650880/1000/self.canvas_size