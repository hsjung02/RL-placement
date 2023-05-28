# -*- coding: utf-8 -*-
from math import sqrt
import numpy as np
import pickle



#global variables==============================================
canvas_grid_x_num = 32
canvas_grid_y_num = 32
canvas_x = 1977172/2000
canvas_y = 1410022/2000
#canvas_x = 2716400/1000
#canvas_y = 2650880/1000
grid_width =canvas_x/(canvas_grid_x_num) #width unit
grid_height=canvas_y/(canvas_grid_y_num) #height unit

hard_macro_num = 4 #adjacency index 0~15
soft_macro_num = 1000 #adjacency index 16~4815
pin_num = 1211 #adjacency index 4816~6026
std_num = 35973


std_width = 8/grid_width
std_height = 1.71/grid_height

soft_macro_area = std_width * std_height *36
soft_macro_size = sqrt(soft_macro_area)


partition_number = soft_macro_num
#=================================================================

def read_cells(filename):

  #각 macro들의 이름 및 크기 정보 가진 딕셔너리
    hard_macro_info={}
    std_info={}

    #각 macro들의 이름만 가진 배열
    hard_macro_name=[]
    std_name=[]

    macro_read_en=0

    #lef 파일로부터 넷리스트에 포함된 macro들의 정보 읽어오기=========================

    with open(filename + '.lef') as n:

        for num, line in enumerate(n):

            if '#' in line:
                continue

            if 'MACRO' in line: #새로운 macro 종류 등장
                macro_read_en=1 
                ismacro = 0 #1일경우 hard macro, 0일경우 std
                data = line.split()
                macro_name = data[1]

        #macro 정보가 아닌 곳에서 size읽어오면 안되므로 enable 신호 사용
            if macro_read_en: 
            #macro가 hard macro인지 std인지 분류====================
                if 'CLASS' in line: 
                    data = line.split()
                    if(data[1]=='BLOCK'):
                        ismacro = 1 #hard macro
                    else:
                        ismacro = 0 #standard cell
            #=========================================================
                elif 'BY' in line:
                    data = line.split()
                    macro_width = data[1]
                    macro_height = data[3]

                    if(ismacro): #hard macro일 경우
                        hard_macro_info[macro_name] = {'width' : float(macro_width)/grid_width, 'height' : float(macro_height)/grid_height}
                        hard_macro_name.append(macro_name)

                    else: #std일 경우
                        try:
                            std_info[macro_name] =  {'width' : float(macro_width)/grid_width, 'height' : float(macro_height)/grid_height}
                        except ValueError:
                            std_info[macro_name] = {'width' : float(data[2])/grid_width, 'height' : float(data[4])/grid_height}

                        std_name.append(macro_name)

    #=====================================================================================


    #def파일로부터 핀 읽어오기============================================================
    
    pins={}
    read_pin_en=0
    pin_adjacency_index = hard_macro_num + soft_macro_num
    with open(filename + '.def') as n:
        for num, line in enumerate(n):
        #pin read enable==================
            if 'END PINS' in line:
                read_pin_en = 0
            elif 'PINS' in line:
                read_pin_en = 1
        #==================================
        
        
            if(read_pin_en):
                if 'pin' in line:
                    data = line.split()
                    pin_name = data[1] #pin의 이름
                    pin_connected_net = data[4] #pin이 어떤 net에 연결되어 있는지
                elif 'PLACED' in line:
                    data = line.split()

                    #um단위
                    pin_x = int(data[3])/2000
                    pin_y = int(data[4])/2000

                    #(0 0) ( 2716400 0 ) ( 0 2650880 ) ( 2716400 2650880 )와 같이 꼭짓점에는 핀이 없당

                    #핀리스트에 추가
                    pins[pin_name] = {'connected_net' : pin_connected_net, 'adjacency_index':pin_adjacency_index, 'x':pin_x/grid_width, 'y':pin_y/grid_height}
                    pin_adjacency_index += 1



  #=====================================================================================


        
  #def 파일로부터 component의 이름 읽어오기=======================================

    hard_macros = {}
    stds = {}

    #initialize
    hard_macro_adjacency_index = 0
    
    read_component_en=0
    std_hmetis_index=1 

    with open(filename + '.def') as n:
        for num, line in enumerate(n):
        #component read enable=============
            if 'END COMPONENTS' in line:
                read_component_en = 0
            elif 'COMPONENTS' in line:
                read_component_en = 1
            #==================================

            if(read_component_en):
                if('- ') in line:
                    data = line.split()
                    component_name = data[1] #각 component의 이름
                    component_macro_name = data[2] #각 component가 어떤 매크로인지지


                    #component_macro_name이 hard_macro_name에 속해있는 경우
                    if component_macro_name in hard_macro_name:
                        hard_macros[component_name] = {'connected_nets' : [], 'adjacency_index':hard_macro_adjacency_index, 'width':hard_macro_info[component_macro_name]['width'], 'height':hard_macro_info[component_macro_name]['height']}
                        hard_macro_adjacency_index += 1
                    #component_macro_name이 std_name에 속해있는 경우
                    elif component_macro_name in std_name:
                        stds[component_name] = {'connected_nets' : [], 'hmetis_index': std_hmetis_index}
                        std_hmetis_index += 1


    return hard_macros, stds, pins, hard_macro_name, std_name


def read_nets(filename, hard_macros, stds, pins, hard_macro_name, std_name):

    read_net_en=0
    net_list = {}
    
    with open(filename + '.def') as n:
        for num, line in enumerate(n):

            #net read enable===================
            if 'END NETS' in line:
                read_net_en = 0
            elif 'NETS' in line:
                read_net_en = 1
            #==================================

            if(read_net_en):
                #새로운 net 등장
                if '-' in line:
                    data = line.split()
                    net_name = data[1] #net의 이름
                    
                    net_list[net_name] = {'connected_stds': [], 'connected_hard_macros' :[], 'connected_soft_macros' :[],'connected_pins' :[], 'connected_adjacency_indices':[]}
                
                else:
                    data = line.split()
                    #해당 net에 포함된 pin 및 component들 추가====
                    for component_name in data: 
                        if 'pin' in component_name: #component가 pin일 경우
                            net_list[net_name]['connected_pins'].append(component_name)
                            net_list[net_name]['connected_adjacency_indices'].append(pins[component_name]['adjacency_index'])

                        elif 'inst' in component_name:
                            if component_name in hard_macros: #component가 hard macro인 경우
                                hard_macros[component_name]['connected_nets'].append(net_name)  #cell list 내의 net update
                                net_list[net_name]['connected_hard_macros'].append(component_name)  #net list update
                                net_list[net_name]['connected_adjacency_indices'].append(hard_macros[component_name]['adjacency_index'])

                            elif component_name in stds: #component가 std인 경우
                                stds[component_name]['connected_nets'].append(net_name)  #cell list 내의 net update
                                net_list[net_name]['connected_stds'].append(component_name)  #net list update

                #==============================================
  

    return net_list


def make_HGraphFile(filename, net_list, stds):

    f=open("./netlist/HGraphFile.txt", 'w')

    #write the information of graph at the top
    info_data = "%d %d\n" % (len(net_list), len(stds)+1)
    f.write(info_data) 

    for net_name in net_list:
        data=""
        for std_name in net_list[net_name]['connected_stds']:
            data = data + str(stds[std_name]['hmetis_index']) + ' '
            data = data+'\n'
            if(data =='\n'):
                data = "%d\n" % (len(stds)+1)  #hard macro만 포함된  net의 모든 hard macro는 191988번째 std로 가정(hmetis 오류 방지용)
        
        f.write(data)
        
    f.close()

def make_softmacros(net_list, partition_num, stds, hard_macros):
  
    #빠른 서치를 위해 hmetis_indices 만들기==============================
    hmetis_indices_std_name = [i for i in range(len(stds)+1)]
    
    i=1
    for std_name in stds:
        if(stds[std_name]['hmetis_index']==i):
            hmetis_indices_std_name[i] = std_name
            i += 1
        
    #===================================================================
    
    #soft_macros 딕셔너리 만들기=========================================
    soft_macros = {}
    soft_macro_count = 0
    soft_macro_adjacency_index = hard_macro_num
    
    print("making soft macros start!")
    with open('./netlist/HGraphFile.hgr.part.'+str(partition_num)) as n:

        for num,line in enumerate(n):
            data = line.split()
            soft_macro_hmetis_name = data[0]

            #Hmetis 시 마지막에 가상의 std cell 넣어두었으므로 실제 std cell이 아니면 읽어오지 않는다
            if(num>=std_num):
                break
            
            else:
                #hmetis 결과 안에 각 std가 몇 번째 softmacro인지 작성되어 있으므로 이 번호를 이용해 소프트매크로의 이름 만들기
                soft_macro_name = 'softmacro' + soft_macro_hmetis_name 
                
                if(soft_macro_name in soft_macros):
                    #num번째(num은 0부터 시작) 줄에 작성된 정보는 num+1번 hmetis index를 갖는 std cell이 속한 soft macro를 나타낸다. 
                    std_hmetis_index = num+1
                
                    #소프트매크로 안에 포함된 스탠다드 셀들
                    soft_macros[soft_macro_name]['clustered_stds'].append(hmetis_indices_std_name[std_hmetis_index])

                    #소프트매크로 안에 포함된 스탠다드 셀들 정보를 이용해 connected net 정보 추가
                    for std_name in soft_macros[soft_macro_name]['clustered_stds']:
                        soft_macros[soft_macro_name]['connected_nets'] = soft_macros[soft_macro_name]['connected_nets'] + stds[std_name]['connected_nets']

                
                
                #처음 등장한 softmacro일 경우
                else:
                    soft_macros[soft_macro_name] = {'connected_nets': [], 'adjacency_index': soft_macro_adjacency_index, 'clustered_stds':[], 'width': soft_macro_size, 'height': soft_macro_size}
                    soft_macro_count += 1
                    soft_macro_adjacency_index += 1
        #=================================================================================
    
    
    #만들어진 소프트 매크로들의 정보를 넷리스트에도 추가하기===============================
    for soft_macro_name in soft_macros:
        for net_name in soft_macros[soft_macro_name]['connected_nets']:
            
            if(soft_macro_name in net_list[net_name]['connected_soft_macros']):
                continue
            else:
                net_list[net_name]['connected_soft_macros'].append(soft_macro_name)
                
            if(soft_macros[soft_macro_name]['adjacency_index'] in net_list[net_name]['connected_adjacency_indices']):
                continue
            else:
                net_list[net_name]['connected_adjacency_indices'].append(soft_macros[soft_macro_name]['adjacency_index'])
            
        '''  
        if(soft_macros[soft_macro_name]['adjacency_index'] in net_list[net_name]['connected_adjacency_indices']):
            continue
        else:
            net_list[net_name]['connected_adjacency_indices'].append(soft_macros[soft_macro_name]['adjacency_index'])
        '''  
    #==================================================================================


    print("making soft macros finish!")
    
    return soft_macros

def make_adjacency_matrix(net_list, total_data_num):
  

    adjacency_matrix = [[0 for i in range(total_data_num)] for j in range(total_data_num)]
  
    print("making adjacency matrix start!")
    
    count=0
    
    for net_name in net_list:
        for index_x in net_list[net_name]['connected_adjacency_indices']:
            for index_y in net_list[net_name]['connected_adjacency_indices']:
                adjacency_matrix[index_x][index_y] = 1


    for i in range(len(adjacency_matrix)):
        adjacency_matrix[i][i] = 0
  
    return adjacency_matrix

def load_netlist(path="./netlist"):
    with open(path+"/adjacency_matrix", "rb") as f:
        adjacency_matrix = pickle.load(f)

    with open(path+"/cells", "rb") as f:
        cells = pickle.load(f)

    with open(path+"/macro_indices", "rb") as f:
        macro_indices = pickle.load(f)

    with open(path+"/std_indices", "rb") as f:
        std_indices = pickle.load(f)
    
    with open(path+"/pin_indices", "rb") as f:
        pin_indices = pickle.load(f)

    return adjacency_matrix, cells, macro_indices, std_indices, pin_indices

if __name__ == "__main__":

    filename = './netlist/ispd18_test3'

    #before clustering==========
    hard_macros, stds, pins, hard_macro_name, std_name = read_cells(filename)
    net_list = read_nets(filename, hard_macros, stds, pins, hard_macro_name, std_name)
    make_HGraphFile(filename, net_list,stds)
    print("parsing finish!")


    #after clustering=========
    soft_macros = make_softmacros(net_list, partition_number, stds, hard_macros)
    total_data_num = len(hard_macros)+len(soft_macros)+len(pins)
    adjacency_matrix = make_adjacency_matrix(net_list, total_data_num)

    hm = {hard_macros[c]['adjacency_index']:{'width':hard_macros[c]['width'], 'height':hard_macros[c]['height']} for c in hard_macros}
    sm = {soft_macros[c]['adjacency_index']:{'width':soft_macro_size, 'height':soft_macro_size} for c in soft_macros}
    p = {pins[c]['adjacency_index']:{'x':pins[c]['x'], 'y':pins[c]['y']} for c in pins}
    cells = {}
    cells.update(hm)
    cells.update(sm)
    cells.update(p)
    #cells = hm | sm | p

    macro_indices = [hard_macros[c]['adjacency_index'] for c in hard_macros]
    std_indices = [soft_macros[c]['adjacency_index'] for c in soft_macros]
    pin_indices = [pins[c]['adjacency_index'] for c in pins]


    print("making adjacency matrix finish!")

    path = "./netlist"
    with open(path+"/adjacency_matrix", "wb") as f:
        pickle.dump(adjacency_matrix, f)

    with open(path+"/cells", "wb") as f:
        pickle.dump(cells, f)

    with open(path+"/macro_indices", "wb") as f:
        pickle.dump(macro_indices, f)

    with open(path+"/std_indices", "wb") as f:
        pickle.dump(std_indices, f)
    
    with open(path+"/pin_indices", "wb") as f:
        pickle.dump(pin_indices, f)