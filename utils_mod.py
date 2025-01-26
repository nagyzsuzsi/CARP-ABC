import numpy as np
import random
import math
import copy

from itertools import combinations
from numpy import random as npr
from operator import attrgetter
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import networkx as nx

inf = 2100000000

# HMA
alpha = 0.6

# ACOPR
Iter = 1500  # number of iterations
beta = 0.8  # used in the state transition rule
rho = 0.1   # used in pheromone calculation
q_0 = 0.9   # used in the state transition rule
tau_0 = 0.2 # initial pheromone value
b = 10      # number of ants?

class Arc:
    
    def __init__(self, head_node, tail_node, trav_cost):
        self.head_node = head_node
        self.tail_node = tail_node
        self.trav_cost = trav_cost
    
    def __str__(self):
        return "("+str(self.head_node)+","+str(self.tail_node)+")\ttc:"+str(self.trav_cost)

    def __repr__(self):
        return "("+str(self.head_node)+","+str(self.tail_node)+")\tc:"+str(self.trav_cost)


class Task(Arc):
    
    def __init__(self, head_node, tail_node, trav_cost, serv_cost, demand, inverse):
        super(Task, self).__init__(head_node, tail_node, trav_cost)
        self.id = 0
        self.serv_cost = serv_cost
        self.demand = demand
        self.inverse = inverse

    def __str__(self):
        return str(self.id)+"\t("+str(self.head_node)+","+str(self.tail_node)+")\ttc: "+str(self.trav_cost) \
            +"\tsc: "+str(self.serv_cost)+"\td: "+str(self.demand)+"\tinv: "+str(self.inverse)+"\n"

    def __repr__(self):
        return str(self.id)+"\t("+str(self.head_node)+","+str(self.tail_node)+")\ttc: "+str(self.trav_cost) \
            +"\tsc: "+str(self.serv_cost)+"\td: "+str(self.demand)+"\tinv: "+str(self.inverse)+"\n"


class Individual:

    def __init__(self):
        self.index = 0
        self.sequence = None
        self.route_seg = None         # (start_id, end_id)
        self.route_seg_load = None
        self.total_cost = inf
        self.fitness = 0.0
        self.age = 0

class Instance:
    
    def __init__(self):
        self.name = None
        self.vertex_num = 0
        self.req_edge_num = 0
        self.nonreq_edge_num = 0
        self.vehicle_num = 0
        self.capacity = 0
        self.depot = None
        
        self.solution_lb = 0
        self.task_num = 0
        self.total_arc_num = 0
        
        self.virtual_task_ids = list() # for every route it contains its current virtual task id
        self.not_allowed_vt_ids = set()
        self.task_ids = None  # tasks that still need to be served
        self.task_ids_with_inv = None
        
        self.tasks = None
        self.arcs = None
        
        self.finished_routes = set()
        self.free_vehicles = set()
        self.route_to_vehicle = None
        self.last_processed_event_id = -1
        
        self.start_time = 0
    

    def ind_from_seq(self, seq):
        ind = Individual()
        ind.sequence = seq
        ind.route_seg = self.find_route_segments(ind.sequence)
        ind.route_seg_load = self.calculate_route_segments_load(ind.sequence, ind.route_seg)
        ind.total_cost = self.calculate_tc(ind.sequence)
        ind.fitness = self.caculate_fitness(ind.total_cost)
        return ind

    
    def fix_sequence(self, seq):
        fixed_seq = list()
        for x in range(len(seq)-1):
            if seq[x] == 0 and seq[x+1] != 0:
                fixed_seq.append(0)
            elif seq[x] != 0:
                fixed_seq.append(seq[x])
        fixed_seq.append(seq[-1])
        if fixed_seq[-1] != 0:
            fixed_seq.append(0)
        return fixed_seq
    
    
    def construct_routing_plan(self, seq):
        routing_plan = list()
        for i in range(len(seq)):
            if seq[i] != 0:
                routing_plan.append( (self.tasks[seq[i]].head_node, self.tasks[seq[i]].tail_node) )
        return routing_plan      

    
    def construct_whole_routing_plan(self, seq):
        routing_plan = list()
        for i in range(len(seq)-1):
            for j in range(1,self.shortest_path[self.tasks[seq[i]].tail_node][self.tasks[seq[i+1]].head_node][0]+1):
                routing_plan.append(self.shortest_path[self.tasks[seq[i]].tail_node][self.tasks[seq[i+1]].head_node][j])
        return routing_plan
    

    def find_arc_id(self, head_node, tail_node):
        return next((arc_id for arc_id, arc in enumerate(self.arcs) \
                if arc.head_node == head_node and arc.tail_node == tail_node), -1)
    
    
    def find_task_id(self, head_node, tail_node):
        return next((task_id for task_id, task in enumerate(self.tasks) \
                if task != None and task.head_node == head_node and task.tail_node == tail_node), -1)
    

    # find the route segments (i.e., the start and end tasks of each route)
    def find_route_segments(self, s):
        route_seg = []
        i = 0
        while i < len(s)-1:
            start = i
            i += 1
            while s[i] != 0:
                i +=1
            end = i
            route_seg.append((start, end))
        return(route_seg)
    
    
    # calculate the load of the route segments (if the capacity constraint is broken, return None)
    def calculate_route_segments_load(self, s, route_seg):
        route_seg_load = []
        for r in route_seg:
            load = 0
            for i in range(r[0]+1,r[1]):
                load += self.tasks[s[i]].demand
#            if load > self.capacity:
#                return None
            route_seg_load.append(load)
        return(route_seg_load)
    
    
    # for IDP and to check if a solution is feasible
    def calculate_excess_demand(self, route_seg_load):
        excess_demand = 0
        for load in route_seg_load:
            if load > self.capacity:
                excess_demand += (load - self.capacity)
        return excess_demand
    
    
    # calculate the total cost of a solution
    def calculate_tc(self, s):
        total_cost = 0
        for i in range(len(s)-1):
            total_cost += (self.tasks[s[i]].serv_cost \
                + self.min_cost[self.tasks[s[i]].tail_node][self.tasks[s[i+1]].head_node])
        return total_cost

    
    # calculate fitness
    def caculate_fitness(self, total_cost):
        return self.solution_lb / total_cost


    def get_route_segment_index(self, route_seg, index):
        for i, r in enumerate(route_seg):
            if index < r[1]:
                return i
    
    
    def get_task_ids_from_sequence(self, seq):
        task_ids = set(seq)
        task_ids.remove(0)
        return list(task_ids)
    

    # READ FILE
    def import_from_file(self, file):
        
        f = open(file, "r")
    
        done = False
        while not done:
            x = f.readline()
            
            if " NOMBRE" in x:
                self.name = x.split(" : ")[1]
            
            elif " VERTICES" in x:
                self.vertex_num = int(x.split(" : ")[1])
            
            elif " ARISTAS_REQ" in x:
                self.req_edge_num = int(x.split(" : ")[1])
                
                self.task_num = 2*self.req_edge_num
                self.tasks = [0 for i in range(self.task_num+1)]
                self.task_ids = set(np.arange(1, self.req_edge_num+1, dtype=int))
                self.task_ids_with_inv = self.task_ids.union(set(np.arange(\
                            self.req_edge_num+1, self.req_edge_num*2+1, dtype=int)))
            
            elif " ARISTAS_NOREQ" in x:
                self.nonreq_edge_num = int(x.split(" : ")[1])
                
                self.total_arc_num = self.task_num + 2*self.nonreq_edge_num
                self.arcs = [0 for i in range(self.total_arc_num+1)]
            
            elif " VEHICULOS" in x:
                self.vehicle_num = int(x.split(" : ")[1])
            
            elif " CAPACIDAD" in x:
                self.capacity = int(x.split(" : ")[1])
            
            elif " LISTA_ARISTAS_REQ" in x:
                for e in range(1, self.req_edge_num+1):
                    head_node, tail_node, serv_cost, demand = map(int, f.readline().replace(" ","")\
                        .replace("coste"," ").replace("demanda"," ")\
                        .replace("(","").replace(")","").replace(","," ").replace("\n","").split(" "))
                    self.tasks[e] = Task(head_node, tail_node, serv_cost, serv_cost, demand, e+self.req_edge_num)
                    self.tasks[e].id = e
                    self.tasks[e+self.req_edge_num] = Task(tail_node, head_node, serv_cost, serv_cost, demand, e)
                    self.tasks[e+self.req_edge_num].id = e+self.req_edge_num
                    self.arcs[e] = Arc(head_node, tail_node, serv_cost)
                    self.arcs[e+self.req_edge_num] = Arc(tail_node, head_node, serv_cost)
                    self.solution_lb += self.tasks[e].demand
                
            elif " LISTA_ARISTAS_NOREQ" in x:
                for e in range(self.task_num+1, self.task_num+self.nonreq_edge_num+1):
                    head_node, tail_node, trav_cst = map(int, f.readline().replace(" ","")\
                        .replace("coste"," ")\
                        .replace("(","").replace(")","").replace(","," ").replace("\n","").split(" "))
                    self.arcs[e] = Arc(head_node, tail_node, trav_cst)
                    self.arcs[e+self.nonreq_edge_num] = Arc(tail_node, head_node, trav_cst)
            
            elif " DEPOSITO" in x:
                self.depot = int(x.split(" : ")[1])
                
                self.tasks[0] = Task(self.depot, self.depot, 0, 0, 0, 0)
                self.arcs[0] = Arc(self.depot, self.depot, 0)
                
                done = True
        
        f.close()
        

    # calculate the shortest paths between two nodes (using Dijkstra's shortest path algorithm)
    def calculate_shortest_paths(self):       
        max_node = self.vertex_num + 1
        self.trav_cost = np.full((max_node,max_node), inf, dtype=int)
        self.serve_cost = np.full((max_node,max_node), inf, dtype=int)
        self.shortest_path = np.zeros((max_node,max_node,max_node), dtype=int)
        self.min_cost = np.full((max_node,max_node), inf, dtype=int)
        
        for i in range(1,max_node):
            for j in range(1,max_node):
                self.trav_cost[i][j] = inf;
        
        for i in range(1,self.total_arc_num+1):
            self.trav_cost[self.arcs[i].head_node][self.arcs[i].tail_node] = self.arcs[i].trav_cost
        
        for i in range(1,self.task_num+1):
            self.serve_cost[self.tasks[i].head_node][self.tasks[i].tail_node] = self.tasks[i].serv_cost
        
        for i in range(1,max_node):
            for j in range(1,max_node):
                if j == i:
                    continue
                self.shortest_path[i][j][0] = 1
                self.shortest_path[i][j][1] = i
                self.min_cost[i][j] = inf
        
        mark = np.zeros((max_node,), dtype=int)
        dist = np.zeros((max_node,), dtype=int)
        dist1 = np.zeros((max_node,), dtype=int)
        nearest_neighbor = np.zeros((max_node,), dtype=int)
        
        for i in range(1,max_node):
            mark[i] = 1
            
            for j in range(1,max_node):
                if j == i:
                    continue
                mark[j] = 0
                dist[j] = self.trav_cost[i][j]
                dist1[j] = dist[j]
            
            for k in range(1,self.vertex_num):
                minimum = inf
                nearest_neighbor[0] = 0     # needed?
                
                for j in range(1,max_node):
                    if mark[j]:
                        continue
                    if dist1[j] == inf:
                        continue
                    if dist1[j] < minimum:
                        minimum = dist1[j]
                
                if minimum == inf:
                    continue
                
                for j in range(1,max_node):
                    if mark[j]:
                        continue
                    if dist1[j] == minimum:
                        nearest_neighbor[0] += 1
                        nearest_neighbor[nearest_neighbor[0]] = j
                
                v = nearest_neighbor[1]
                dist1[v] = inf
                mark[v] = 1
                
                if self.shortest_path[i][v][0] == 0 or \
                    (self.shortest_path[i][v][0] > 0 and self.shortest_path[i][v][self.shortest_path[i][v][0]] != v):
                        self.shortest_path[i][v][0] += 1
                        self.shortest_path[i][v][self.shortest_path[i][v][0]] = v
                
                for j in range(1,max_node):
                    
                    if mark[j]:
                        continue
                    
                    if minimum + self.trav_cost[v][j] < dist[j]:
                        dist[j] = minimum + self.trav_cost[v][j]
                        dist1[j] = minimum + self.trav_cost[v][j]
                        for m in range(self.shortest_path[i][v][0]+1):
                            self.shortest_path[i][j][m] = self.shortest_path[i][v][m]
                
                for j in range(1,max_node):
                    if j == i:
                        continue
                    self.min_cost[i][j] = dist[j]
        
        for i in range(1,max_node):
            for j in range(1,max_node):
                if self.shortest_path[i][j][0] == 1:
                    self.shortest_path[i][j][0] = 0
        
        for i in range(1,max_node):
            self.shortest_path[i][i][0] = 1
            self.shortest_path[i][i][1] = i
            self.min_cost[i][i] = 0

    
# ----------------------------- generate solution ----------------------------------

    def random_routing_plan_generator(self, task_ids):
        task_seq = npr.permutation(list(task_ids))
        ind_seq = [0]
        
        route_load = 0
        for i in range(len(task_seq)):
            inv = random.randint(0,1)
            route_load += self.tasks[task_seq[i]].demand
            if route_load > self.capacity:
                route_load = self.tasks[task_seq[i]].demand
                ind_seq.append(0)
            if inv and self.tasks[task_seq[i]].inverse != None:
                ind_seq.append(self.tasks[task_seq[i]].inverse)
            else:
                ind_seq.append(task_seq[i])
        ind_seq.append(0)
        
        ind = Individual()
        ind.sequence = ind_seq
        ind.route_seg = self.find_route_segments(ind.sequence)
        ind.route_seg_load = self.calculate_route_segments_load(ind.sequence, ind.route_seg)       
        ind.total_cost = self.calculate_tc(ind.sequence)
        ind.fitness = self.caculate_fitness(ind.total_cost)
        ind.age = 0
        
        return ind

    
    def randomized_path_scanning_heuristic(self, task_ids):
        sequences = [[0] for i in range(5)]
        inds = [0 for i in range(5)]
        
        for i in range(5):
            unserved_task_ids = copy.deepcopy(task_ids)
            current_route_load = 0
            while len(unserved_task_ids) != 0:
                servable_task_ids = list(filter(lambda task_id: \
                    current_route_load + self.tasks[task_id].demand <= self.capacity, unserved_task_ids))
                if len(servable_task_ids) == 0:
                    sequences[i].append(0)
                    current_route_load = 0
                elif len(servable_task_ids) == 1:
                    sequences[i].append(servable_task_ids[0])
                    unserved_task_ids.remove(servable_task_ids[0])
                    if self.tasks[servable_task_ids[0]].inverse != None:
                        unserved_task_ids.remove(self.tasks[servable_task_ids[0]].inverse)
                else:
                    min_cost_task_ids = []
                    min_cost = inf
                    for task_id in servable_task_ids:
                        cost = self.min_cost[self.tasks[sequences[i][-1]].tail_node][self.tasks[task_id].head_node]
                        if cost < min_cost:
                            min_cost = cost
                            min_cost_task_ids = [task_id]
                        if cost == min_cost:
                            min_cost_task_ids.append(task_id)
                    if len(min_cost_task_ids) == 1:
                        sequences[i].append(min_cost_task_ids[0])
                        current_route_load += self.tasks[min_cost_task_ids[0]].demand
                        unserved_task_ids.remove(min_cost_task_ids[0])
                        if self.tasks[min_cost_task_ids[0]].inverse != None:
                            unserved_task_ids.remove(self.tasks[min_cost_task_ids[0]].inverse)                        
                    else:
                        best_value_task_ids = []
                        best_value = None
                        # 1) maximize the distance from the head of task to the depot
                        if i == 0:
                            best_value = 0
                            for task_id in min_cost_task_ids:
                                value = self.min_cost[self.tasks[task_id].head_node][self.tasks[0].head_node]
                                if best_value != None and value > best_value:
                                    best_value = value
                                    best_value_task_ids = [task_id]
                                elif value == best_value:
                                    best_value_task_ids.append(task_id)
                        # 2) minimize the distance from the head of task to the depot
                        if i == 1:
                            best_value = inf
                            for task_id in min_cost_task_ids:
                                value = self.min_cost[self.tasks[task_id].head_node][self.tasks[0].head_node]
                                if best_value != None and value < best_value:
                                    best_value = value
                                    best_value_task_ids = [task_id]
                                elif value == best_value:
                                    best_value_task_ids.append(task_id)
                        # 3) maximize the term dem(t)/sc(t)
                        if i == 2:
                            best_value = 0
                            for task_id in min_cost_task_ids:
                                value = self.tasks[task_id].demand / self.tasks[task_id].serv_cost \
                                    if self.tasks[task_id].serv_cost != 0\
                                    else self.tasks[task_id].demand
                                if best_value != None and value > best_value:
                                    best_value = value
                                    best_value_task_ids = [task_id]
                                elif value == best_value:
                                    best_value_task_ids.append(task_id)                        
                        # 4) minimize the term dem(t)/sc(t)
                        if i == 3:
                            best_value = inf
                            for task_id in min_cost_task_ids:
                                value = self.tasks[task_id].demand / self.tasks[task_id].serv_cost \
                                    if self.tasks[task_id].serv_cost != 0\
                                    else self.tasks[task_id].demand
                                if best_value != None and value < best_value:
                                    best_value = value
                                    best_value_task_ids = [task_id]
                                elif value == best_value:
                                    best_value_task_ids.append(task_id)   
                        # 5) use rule 1) if the vehicle is less than halffull, otherwise use rule 2)
                        if i == 4:
                            if current_route_load < self.capacity / 2:
                                best_value = 0
                                for task_id in min_cost_task_ids:
                                    value = self.min_cost[self.tasks[task_id].head_node][self.tasks[0].head_node]
                                    if best_value != None and value > best_value:
                                        best_value = value
                                        best_value_task_ids = [task_id]
                                    elif value == best_value:
                                        best_value_task_ids.append(task_id)                                
                            else:
                                best_value = inf
                                for task_id in min_cost_task_ids:
                                    value = self.min_cost[self.tasks[task_id].head_node][self.tasks[0].head_node]
                                    if best_value != None and value < best_value:
                                        best_value = value
                                        best_value_task_ids = [task_id]
                                    elif value == best_value:
                                        best_value_task_ids.append(task_id)                                
                        
                        next_task_id = None
                        if len(best_value_task_ids) > 1:                       
                            next_task_id = random.choice(min_cost_task_ids)
                        else:
                            next_task_id = min_cost_task_ids[0]
                        sequences[i].append(next_task_id)
                        current_route_load += self.tasks[next_task_id].demand
                        unserved_task_ids.remove(next_task_id)
                        if self.tasks[next_task_id].inverse != None:
                            unserved_task_ids.remove(self.tasks[next_task_id].inverse)
            sequences[i].append(0)
        
        for i in range(5):
            inds[i] = Individual()
            inds[i].sequence = sequences[i]
            inds[i].route_seg = self.find_route_segments(inds[i].sequence)
            inds[i].route_seg_load = self.calculate_route_segments_load(inds[i].sequence, inds[i].route_seg)       
            inds[i].total_cost = self.calculate_tc(inds[i].sequence)
            inds[i].fitness = self.caculate_fitness(inds[i].total_cost)
            #print(inds[i].sequence, inds[i].total_cost)
        
        best_ind = min(inds, key=attrgetter('total_cost'))
        
        return best_ind
    
    
# ----------------------------------- search operators ----------------------------------

 
    def inverse(self, ind, task_id=None):    
        ind_new = copy.deepcopy(ind)
        
        task_ids = self.get_task_ids_from_sequence(ind.sequence)
        
        if task_id == None:            
            task_id = random.sample(task_ids, k=1)[0]
        x = ind.sequence.index(task_id)
        
        if self.tasks[task_id].inverse != None:
            ind_new.sequence[x] = self.tasks[task_id].inverse
        else:
            return ind
        
        ind_new.total_cost = self.calculate_tc(ind_new.sequence)
        
        return ind_new
    

    def insert(self, ind, task_id_1=None, task_id_2=None, only_feasible=True):
        # for inserting task_id_1 before task_id_2
        ind_new1 = copy.deepcopy(ind)   # for case 1: e.g., (a,b),(c,d)
        ind_new2 = copy.deepcopy(ind)   # for case 2: e.g., inv((a,b)),(c,d) = (b,a),(c,d)
        # for inserting task_id_1 after task_id_2
        ind_new3 = copy.deepcopy(ind)   # for case 3: e.g., (c,d),(a,b)
        ind_new4 = copy.deepcopy(ind)   # for case 4: e.g., (c,d),inv((a,b)) = (c,d),(b,a)
        
        task_ids = self.get_task_ids_from_sequence(ind.sequence)

        # select task1
        if task_id_1 == None:            
            task_id_1 = random.sample(task_ids, k=1)[0]
        x = ind.sequence.index(task_id_1)
        
        rs_index = self.get_route_segment_index(ind.route_seg, x)
        
        # remove task1 from its original position
        del(ind_new1.sequence[x])
        
        # if it was the only task in the route plan remove the excess 0
        if ind.route_seg[rs_index][1] - ind.route_seg[rs_index][0] == 2:
            del(ind_new1.sequence[x])

        # insert task1 before/after task2
        ind_new3 = copy.deepcopy(ind_new1)
        if task_id_2 == None:
            task_ids.remove(task_id_1)
            task_id_2 = random.sample(task_ids, k=1)[0]
        x_new = ind_new1.sequence.index(task_id_2)
        ind_new1.sequence.insert(x_new, task_id_1)
        ind_new3.sequence.insert(x_new+1, task_id_1)
        
        # insertion into a new route plan is impossible if the task can be inserted only right before/after another task
        
        try:
            self.find_route_segments(ind_new1.sequence)
        except:
            print("insert error")
            print(ind.sequence, task_id_1, task_id_2)
            print(ind_new1.sequence)
            
            for ind_new in [ind_new1, ind_new2, ind_new3, ind_new4]:
                ind_new.sequence = self.fix_sequence(ind_new.sequence)

        ind_new1.route_seg = self.find_route_segments(ind_new1.sequence)
        ind_new1.route_seg_load = self.calculate_route_segments_load(ind_new1.sequence, ind_new1.route_seg)
        if only_feasible and self.calculate_excess_demand(ind_new1.route_seg_load)>0:
            return ind
        ind_new1.total_cost = self.calculate_tc(ind_new1.sequence)
        
        # create another solutions with inverse
        # before
        ind_new2.sequence = copy.deepcopy(ind_new1.sequence)
        ind_new2.sequence[x_new] = self.tasks[ind_new2.sequence[x_new]].inverse \
            if self.tasks[ind_new2.sequence[x_new]].inverse != None else ind_new2.sequence[x_new]
        # after
        ind_new4.sequence = copy.deepcopy(ind_new3.sequence)
        ind_new4.sequence[x_new+1] = self.tasks[ind_new4.sequence[x_new+1]].inverse \
            if self.tasks[ind_new4.sequence[x_new+1]].inverse != None else ind_new4.sequence[x_new+1]
        
        #print(ind_new1.sequence, ind_new1.total_cost)
        for ind_new in [ind_new2, ind_new3, ind_new4]:
            ind_new.route_seg = copy.deepcopy(ind_new1.route_seg)
            ind_new.route_seg_load = copy.deepcopy(ind_new1.route_seg_load)
            ind_new.total_cost = self.calculate_tc(ind_new.sequence)
            #print(ind_new.sequence, ind_new.total_cost)

        # return the solution with the lowest total cost
        return min([ind_new1, ind_new2, ind_new3, ind_new4], key=attrgetter('total_cost'))
    
    
    def double_insert(self, ind, task_id_1=None, task_id_2=None, only_feasible=True):
        # for inserting task_id_1 and the next task before task_id_2
        ind_new1 = copy.deepcopy(ind)   # for case 1: (a,b),(c,d)
        ind_new2 = copy.deepcopy(ind)   # for case 2: inv((a,b)),(c,d) = (b,a)(c,d)
        ind_new3 = copy.deepcopy(ind)   # for case 3: (a,b),inv((c,d)) = (a,b)(d,c)
        ind_new4 = copy.deepcopy(ind)   # for case 4: inv((a,b)),inv((c,d)) = (b,a)(d,c)
        # for inserting task_id_1 and the next task after task_id_2
        ind_new5 = copy.deepcopy(ind)   # for case 5: (a,b),(c,d)
        ind_new6 = copy.deepcopy(ind)   # for case 6: inv((a,b)),(c,d) = (b,a)(c,d)
        ind_new7 = copy.deepcopy(ind)   # for case 7: (a,b),inv((c,d)) = (a,b)(d,c)
        ind_new8 = copy.deepcopy(ind)   # for case 8: inv((a,b)),inv((c,d)) = (b,a)(d,c)
        
        task_ids = self.get_task_ids_from_sequence(ind.sequence)
        
        # select task1 and next task as the task sequence (len = 2)
        if task_id_1 == None:
            task_id_1 = random.sample(task_ids, k=1)[0]
        x = ind.sequence.index(task_id_1)
        tasks = ind.sequence[x:x+2]

        # if the next task is the depot or the 2nd given taskid is the same as the id of the 2nd task in the sequence
        # then the operation cannot be executed
        if tasks[1] == 0 or tasks[1] == task_id_2:
            return ind

        rs_index = self.get_route_segment_index(ind.route_seg, x)

        # remove the tasks from their original positions
        del(ind_new1.sequence[x])
        del(ind_new1.sequence[x])
        
        # if they were the only tasks in the route plan remove the excess 0
        if ind.route_seg[rs_index][1] - ind.route_seg[rs_index][0] == 3:
            del(ind_new1.sequence[x])
        
        # insert the tasks before/after task2
        ind_new5 = copy.deepcopy(ind_new1)
        if task_id_2 == None:
            task_ids.remove(tasks[0])
            task_ids.remove(tasks[1])
            task_id_2 = random.sample(task_ids, k=1)[0]
        x_new = ind_new1.sequence.index(task_id_2)
        ind_new1.sequence[x_new:x_new] = tasks
        ind_new5.sequence[x_new+1:x_new+1] = tasks
        
        # insertion into a new route plan is impossible if the task can be inserted only right before/after another task

        # before
        ind_new2.sequence = copy.deepcopy(ind_new1.sequence)
        ind_new3.sequence = copy.deepcopy(ind_new1.sequence)
        ind_new4.sequence = copy.deepcopy(ind_new1.sequence)
        
        # after
        ind_new6.sequence = copy.deepcopy(ind_new5.sequence)
        ind_new7.sequence = copy.deepcopy(ind_new5.sequence)
        ind_new8.sequence = copy.deepcopy(ind_new5.sequence)        

        # create other solutions with inverses
        # case 2
        ind_new2.sequence[x_new] = self.tasks[ind_new2.sequence[x_new]].inverse \
            if self.tasks[ind_new2.sequence[x_new]].inverse != None else ind_new2.sequence[x_new]
        # case 3
        ind_new3.sequence[x_new+1] = self.tasks[ind_new3.sequence[x_new+1]].inverse \
            if self.tasks[ind_new3.sequence[x_new+1]].inverse != None else ind_new3.sequence[x_new+1]
        # case 4
        ind_new4.sequence[x_new] = self.tasks[ind_new4.sequence[x_new]].inverse \
            if self.tasks[ind_new4.sequence[x_new]].inverse != None else ind_new4.sequence[x_new]
        ind_new4.sequence[x_new+1] = self.tasks[ind_new4.sequence[x_new+1]].inverse \
            if self.tasks[ind_new4.sequence[x_new+1]].inverse != None else ind_new4.sequence[x_new+1]
        # case 6
        ind_new6.sequence[x_new+1] = self.tasks[ind_new6.sequence[x_new+1]].inverse \
            if self.tasks[ind_new6.sequence[x_new+1]].inverse != None else ind_new6.sequence[x_new+1]
        # case 7
        ind_new7.sequence[x_new+2] = self.tasks[ind_new7.sequence[x_new+2]].inverse \
            if self.tasks[ind_new7.sequence[x_new+2]].inverse != None else ind_new7.sequence[x_new+2]
        # case 8
        ind_new8.sequence[x_new+1] = self.tasks[ind_new8.sequence[x_new+1]].inverse \
            if self.tasks[ind_new8.sequence[x_new+1]].inverse != None else ind_new8.sequence[x_new+1]
        ind_new8.sequence[x_new+2] = self.tasks[ind_new8.sequence[x_new+2]].inverse \
            if self.tasks[ind_new8.sequence[x_new+2]].inverse != None else ind_new8.sequence[x_new+2]
        
        ind_new1.route_seg = self.find_route_segments(ind_new1.sequence)
        ind_new1.route_seg_load = self.calculate_route_segments_load(ind_new1.sequence, ind_new1.route_seg)       
        if only_feasible and self.calculate_excess_demand(ind_new1.route_seg_load)>0:
            return ind
        ind_new1.total_cost = self.calculate_tc(ind_new1.sequence)

        for ind_new in [ind_new2, ind_new3, ind_new4, ind_new5, ind_new6, ind_new7, ind_new8]:
            ind_new.route_seg = copy.deepcopy(ind_new1.route_seg)
            ind_new.route_seg_load = copy.deepcopy(ind_new1.route_seg_load)
            ind_new.total_cost = self.calculate_tc(ind_new.sequence)

        # return the solution with the lowest total cost
        return min([ind_new1, ind_new2, ind_new3, ind_new4, ind_new5, ind_new6, ind_new7, ind_new8], \
                   key=attrgetter('total_cost'))
    
    
    def swap(self, ind, task_id_1=None, task_id_2=None, only_feasible=True):
        ind_new1 = copy.deepcopy(ind)
        ind_new2 = copy.deepcopy(ind)
        ind_new3 = copy.deepcopy(ind)
        ind_new4 = copy.deepcopy(ind)
        
        task_ids = self.get_task_ids_from_sequence(ind.sequence)
        
        if task_id_1 == None:
            task_id_1 = random.sample(task_ids, k=1)[0]      
        x = ind.sequence.index(task_id_1)
        
        if task_id_2 == None:
            task_ids.remove(task_id_1)
            task_id_2 = random.sample(task_ids, k=1)[0] 
        y = ind.sequence.index(task_id_2)
        
        if ind.sequence[x] != ind.sequence[y]:
            inv_x = self.tasks[ind.sequence[x]].inverse \
                if self.tasks[ind.sequence[x]].inverse != None else ind.sequence[x]
            inv_y = self.tasks[ind.sequence[y]].inverse \
                if self.tasks[ind.sequence[y]].inverse != None else ind.sequence[y]
            
            # original - original
            ind_new1.sequence[x] = copy.deepcopy(ind.sequence[y])
            ind_new1.sequence[y] = copy.deepcopy(ind.sequence[x])
            # original - inv
            ind_new2.sequence[x] = copy.deepcopy(ind.sequence[y])
            ind_new2.sequence[y] = inv_x
            # inv - original
            ind_new3.sequence[x] = inv_y
            ind_new3.sequence[y] = copy.deepcopy(ind.sequence[x])
            # inv - inv
            ind_new4.sequence[x] = inv_y
            ind_new4.sequence[y] = inv_x
        else:
            return self.inverse(ind, task_id_1)
        
        try:
            self.find_route_segments(ind_new1.sequence)
        except:
            print("swap error")
            print(ind.sequence, task_id_1, task_id_2)
            print(ind_new1.sequence)
            
            for ind_new in [ind_new1, ind_new2, ind_new3, ind_new4]:
                ind_new.sequence = self.fix_sequence(ind_new.sequence)
                ind_new.route_seg = self.find_route_segments(ind_new.sequence)
        
        ind_new1.route_seg_load = self.calculate_route_segments_load(ind_new1.sequence, ind_new1.route_seg)
        if only_feasible and self.calculate_excess_demand(ind_new1.route_seg_load)>0:
            return ind
        ind_new1.total_cost = self.calculate_tc(ind_new1.sequence)

        for ind_new in [ind_new2, ind_new3, ind_new4]:
            ind_new.route_seg_load = copy.deepcopy(ind_new1.route_seg_load)
            ind_new.total_cost = self.calculate_tc(ind_new.sequence)

        return min([ind_new1, ind_new2, ind_new3, ind_new4], key=attrgetter('total_cost'))
        
    
    def two_opt(self, ind, task_id_1=None, task_id_2=None, only_feasible=True):
        
        task_ids = self.get_task_ids_from_sequence(ind.sequence)
        
        if task_id_1 == None:
            task_id_1 = random.sample(task_ids, k=1)[0]   
        x = ind.sequence.index(task_id_1)    
        x_rid = self.get_route_segment_index(ind.route_seg, x)
        
        if task_id_2 == None:
            task_ids.remove(task_id_1)
            task_id_2 = random.sample(task_ids, k=1)[0] 
        y = ind.sequence.index(task_id_2)
        y_rid = self.get_route_segment_index(ind.route_seg, y)
        
        # single route
        if x_rid == y_rid:
            ind_new = copy.deepcopy(ind)
            
            # normally cannot happen
            if x == y:
                return ind
            
            # if y is before x
            if y < x:
                for i in range(y,x+1):
                    ind_new.sequence[i] = copy.deepcopy(self.tasks[ind.sequence[i]].inverse) \
                        if self.tasks[ind.sequence[i]].inverse != None\
                        else ind_new.sequence[i]
                sub_seq = ind_new.sequence[y:x+1]
                sub_seq.reverse()
                ind_new.sequence[y:x+1] = sub_seq
            # if x is before y
            else:
                for i in range(x,y+1):
                    ind_new.sequence[i] = self.tasks[ind.sequence[i]].inverse \
                        if self.tasks[ind.sequence[i]].inverse != None\
                        else ind.sequence[i]
                sub_seq = ind_new.sequence[x:y+1]
                sub_seq.reverse()
                ind_new.sequence[x:y+1] = sub_seq
            
            ind_new.total_cost = self.calculate_tc(ind_new.sequence)
            ind_new.fitness = self.caculate_fitness(ind_new.total_cost)
            
            return ind_new
            
        # double routes
        else:
            ind_new1 = copy.deepcopy(ind)
            ind_new2 = copy.deepcopy(ind)
            
            # to include the delimiter task on the left side seq
            x = x+1
            y = y+1
            
            x_seq_left = ind.sequence[ind.route_seg[x_rid][0]+1:x]            
            x_seq_right = ind.sequence[x:ind.route_seg[x_rid][1]]
            y_seq_left = ind.sequence[ind.route_seg[y_rid][0]+1:y]
            y_seq_right = ind.sequence[y:ind.route_seg[y_rid][1]]
            
            x_seq_right_rev = ind.sequence[x:ind.route_seg[x_rid][1]]
            for i in range(len(x_seq_right_rev)):
                x_seq_right_rev[i] = self.tasks[x_seq_right[i]].inverse \
                    if self.tasks[x_seq_right[i]].inverse != None else x_seq_right_rev[i]
            x_seq_right_rev.reverse()
            
            y_seq_left_rev = ind.sequence[ind.route_seg[y_rid][0]+1:y]          
            for i in range(len(y_seq_left_rev)):
                y_seq_left_rev[i] = self.tasks[y_seq_left[i]].inverse \
                    if self.tasks[y_seq_left[i]].inverse != None else y_seq_left_rev[i]
            y_seq_left_rev.reverse()
            
            x_rid_seq1 = x_seq_left + y_seq_right            
            y_rid_seq1 = y_seq_left + x_seq_right
            if y_rid < x_rid:          
                if (len(x_rid_seq1) != 0):
                    ind_new1.sequence[ind.route_seg[x_rid][0]+1:ind.route_seg[x_rid][1]] = x_rid_seq1
                else:
                    del(ind_new1.sequence[ind.route_seg[x_rid][0]:ind.route_seg[x_rid][1]])
                if (len(y_rid_seq1) != 0):
                    ind_new1.sequence[ind.route_seg[y_rid][0]+1:ind.route_seg[y_rid][1]] = y_rid_seq1
                else:
                    del(ind_new1.sequence[ind.route_seg[y_rid][0]:ind.route_seg[y_rid][1]])
            else:
                if (len(y_rid_seq1) != 0):
                    ind_new1.sequence[ind.route_seg[y_rid][0]+1:ind.route_seg[y_rid][1]] = y_rid_seq1
                else:
                    del(ind_new1.sequence[ind.route_seg[y_rid][0]:ind.route_seg[y_rid][1]])
                if (len(x_rid_seq1) != 0):
                    ind_new1.sequence[ind.route_seg[x_rid][0]+1:ind.route_seg[x_rid][1]] = x_rid_seq1
                else:
                    del(ind_new1.sequence[ind.route_seg[x_rid][0]:ind.route_seg[x_rid][1]])                
            
            # originally not in HMA!
            x_rid_seq2 = x_seq_left + y_seq_left_rev            
            y_rid_seq2 = x_seq_right_rev + y_seq_right
            if y_rid < x_rid: 
                if (len(x_rid_seq2) != 0):
                    ind_new2.sequence[ind.route_seg[x_rid][0]+1:ind.route_seg[x_rid][1]] = x_rid_seq2
                else:
                    del(ind_new2.sequence[ind.route_seg[x_rid][0]:ind.route_seg[x_rid][1]])
                if (len(y_rid_seq2) != 0):
                    ind_new2.sequence[ind.route_seg[y_rid][0]+1:ind.route_seg[y_rid][1]] = y_rid_seq2
                else:
                    del(ind_new2.sequence[ind.route_seg[y_rid][0]:ind.route_seg[y_rid][1]])
            else:
                if (len(y_rid_seq2) != 0):
                    ind_new2.sequence[ind.route_seg[y_rid][0]+1:ind.route_seg[y_rid][1]] = y_rid_seq2
                else:
                    del(ind_new2.sequence[ind.route_seg[y_rid][0]:ind.route_seg[y_rid][1]])
                if (len(x_rid_seq2) != 0):
                    ind_new2.sequence[ind.route_seg[x_rid][0]+1:ind.route_seg[x_rid][1]] = x_rid_seq2
                else:
                    del(ind_new2.sequence[ind.route_seg[x_rid][0]:ind.route_seg[x_rid][1]])                
            
            
            try:
                self.find_route_segments(ind_new1.sequence)
            except:
                print("2-opt error")
                print(ind.sequence, task_id_1, task_id_2)
                
                print(x_seq_left, x_seq_right, y_seq_left, y_seq_right)
                print(x_seq_right_rev, y_seq_left_rev)
                
                print(ind_new1.sequence)
                print(ind_new2.sequence)
                
                for ind_new in [ind_new1, ind_new2]:
                    ind_new.sequence = self.fix_sequence(ind_new.sequence)
            
            
            ind_new1.route_seg = self.find_route_segments(ind_new1.sequence)
            ind_new1.route_seg_load = self.calculate_route_segments_load(ind_new1.sequence, ind_new1.route_seg)
            if only_feasible and self.calculate_excess_demand(ind_new1.route_seg_load)>0:
                ind_new1 = Individual()
            else:
                ind_new1.total_cost = self.calculate_tc(ind_new1.sequence)
                ind_new1.fitness = self.caculate_fitness(ind_new1.total_cost)
            
            ind_new2.route_seg = self.find_route_segments(ind_new2.sequence)
            ind_new2.route_seg_load = self.calculate_route_segments_load(ind_new2.sequence, ind_new2.route_seg)
            if only_feasible and self.calculate_excess_demand(ind_new2.route_seg_load)>0:
                ind_new2 = Individual()
            else:
                ind_new2.total_cost = self.calculate_tc(ind_new2.sequence)
                ind_new2.fitness = self.caculate_fitness(ind_new2.total_cost) 
            
            ind_new = min([ind_new1, ind_new2], key=attrgetter('total_cost'))
            
            if ind_new.total_cost == inf:
                return ind
            else:
                return ind_new
    
    def greedy_sub_tour_mutation(self, ind, selected_route_id=None):
        ind_new = copy.deepcopy(ind)
        
        if selected_route_id == None:
            selected_route_id = random.randint(0,len(ind.route_seg)-1)
        
        # route without 0s on both sides
        route = ind_new.sequence[ind.route_seg[selected_route_id][0]+1:ind.route_seg[selected_route_id][1]]
        
        if len(route) < 4:
            return ind
        
        # fix values
        l_min = 2
        l_max = max(4, int(math.sqrt(len(route))))
        p_rc = 0.5
        p_cp = 0.8
        p_l = 0.2
        nl_max = 5
        
        # random values
        l = random.randint(l_min, l_max)
        sr_1 = random.randint(0, len(route)-l)
        sr_2 = sr_1 + l - 1
        rnd = random.uniform(0,1)
        
        subroute = route[sr_1:sr_2+1]
        route_wo_subroute = copy.deepcopy(route)
        route_wo_subroute[sr_1:sr_2+1] = []
        route_wo_subroute_ = [0] + route_wo_subroute + [0]
        new_route = None
        
        #print(route, subroute, route_wo_subroute)
        
        # greedy mutation
        if rnd <= p_rc:
            best_i_cost_diff = inf
            best_i = -1
            for i in range(1,len(route_wo_subroute_)-1):
                cost_diff = self.min_cost[self.tasks[route_wo_subroute_[i-1]].tail_node][self.tasks[subroute[0]].head_node] \
                    + self.min_cost[self.tasks[subroute[-1]].tail_node][self.tasks[route_wo_subroute_[i]].head_node]
                if cost_diff < best_i_cost_diff:
                    best_i_cost_diff = cost_diff
                    best_i = i
            new_route = route_wo_subroute_[:best_i] + subroute + route_wo_subroute_[best_i:]
        else:
            # random distortion
            if rnd <= p_cp:
                w = sr_1
                new_route = route_wo_subroute
                while len(subroute) != 0:
                    rnd = random.uniform(0,1)
                    added_task = None
                    # random task
                    if rnd <= p_l:
                        added_task = random.sample(subroute, k=1)[0]
                    # (currently) last task
                    else:
                        added_task = subroute[-1]
                    new_route.insert(w, added_task)
                    subroute.remove(added_task)
                    w += 1
                new_route = [0] + new_route + [0]
            # rotation of sub-route
            else:
                new_route = route
                l_side_tasks = copy.deepcopy(route[:sr_1])
                r_side_tasks = copy.deepcopy(route[sr_2+1:])
                
                if len(l_side_tasks) > nl_max:
                    l_side_tasks_dists = [self.min_cost[self.tasks[l_side_task].head_node][self.tasks[subroute[0]].head_node] for l_side_task in l_side_tasks]
                    l_side_tasks = [task for _,task in sorted(zip(l_side_tasks_dists, l_side_tasks))][:5]
                
                if len(r_side_tasks) > nl_max:
                    r_side_tasks_dists = [self.min_cost[self.tasks[subroute[-1]].tail_node][self.tasks[r_side_task].tail_node] for r_side_task in r_side_tasks]
                    r_side_tasks = [task for _,task in sorted(zip(r_side_tasks_dists, r_side_tasks))][:5]
                
                nl_r1 = route.index(random.sample(l_side_tasks, k=1)[0]) if len(l_side_tasks) > 0 else sr_1
                nl_r2 = route.index(random.sample(r_side_tasks, k=1)[0]) if len(r_side_tasks) > 0 else sr_2
                
                new_route[nl_r1:sr_1] = reversed(route[nl_r1:sr_1])
                new_route[sr_2+1:nl_r2+1] = reversed(route[sr_2+1:nl_r2+1])
                
                for i in range(len(new_route)):
                    if (new_route[i] in new_route[nl_r1:sr_1] or new_route[i] in new_route[sr_2+1:nl_r2+1]) and self.tasks[new_route[i]].inverse != None:
                        new_route[i] = self.tasks[new_route[i]].inverse
                
                new_route = [0] + new_route + [0]
                        
        ind_new.sequence[ind.route_seg[selected_route_id][0]:ind.route_seg[selected_route_id][1]+1] = new_route
        ind_new.total_cost = self.calculate_tc(ind_new.sequence)
        ind_new.fitness = self.caculate_fitness(ind_new.total_cost)
        
        return ind_new
        
    
    def merge_split(self, ind, selected_route_ids=None):
        ind_new = copy.deepcopy(ind)
        
        if selected_route_ids == None:
            p = random.randint(1,len(ind.route_seg)) # number of routes that are selected
            selected_route_ids = random.sample(list(np.arange(len(ind.route_seg), dtype=int)), k=p)
        unserved_tasks = list()
        new_sequence = [0]
        
        for route_id in range(len(ind.route_seg)):
            if route_id in selected_route_ids:
                for seq_pos in range(ind.route_seg[route_id][0]+1,ind.route_seg[route_id][1]):
                    unserved_tasks.append(ind.sequence[seq_pos])
                    if self.tasks[ind.sequence[seq_pos]].inverse != None:
                        unserved_tasks.append(self.tasks[ind.sequence[seq_pos]].inverse)
            else:
                new_sequence += ind.sequence[ind.route_seg[route_id][0]+1:ind.route_seg[route_id][1]+1]
        
        new_sequence += self.randomized_path_scanning_heuristic(unserved_tasks).sequence[1:]
        
        ind_new.sequence = new_sequence
        ind_new.route_seg = self.find_route_segments(ind_new.sequence)
        ind_new.route_seg_load = self.calculate_route_segments_load(ind_new.sequence, ind_new.route_seg)
        ind_new.total_cost = self.calculate_tc(ind_new.sequence)
        ind_new.fitness = self.caculate_fitness(ind_new.total_cost)
        
        return ind_new
 

# ----------------------------- DCARP components -----------------------------------


    def prepare_tasks(self):
        self.not_allowed_vt_ids = set()
        for r_id, vt_id in enumerate(self.virtual_task_ids):
            if r_id in self.finished_routes:
                self.not_allowed_vt_ids.add(vt_id)
        
        #print(self.not_allowed_vt_ids)
        #print(self.tasks)
        
        for vt_id in self.not_allowed_vt_ids:
            self.task_ids.remove(vt_id)
            self.task_ids_with_inv.remove(vt_id)
    
    
    def add_finished_routes(self, seq):
        for vt_id in self.not_allowed_vt_ids:
            seq.extend([vt_id, 0])
        return self.ind_from_seq(seq)
        
    
    def remove_finished_routes(self, seq):
        for vt_id in self.not_allowed_vt_ids:
            vt_id_index = seq.index(vt_id)
            seq[vt_id_index:vt_id_index+2] = []
        return self.ind_from_seq(seq)
    
    
    def generate_travel_service_log(self, ind, min_r_cost=0, max_r_cost=inf, stopping_r_cost=None):
        
        seq = ind.sequence      
        
        if max_r_cost == inf:
            max_r_cost = 0
            for r_id in range(len(ind.route_seg)):
                #print(ind.sequence[ind.route_seg[r_id][0]:ind.route_seg[r_id][1]+1])
                r_i_cost = self.calculate_tc(seq[ind.route_seg[r_id][0]:ind.route_seg[r_id][1]+1])
                if r_i_cost > max_r_cost:
                    max_r_cost = r_i_cost 
        if stopping_r_cost == None:
            stopping_r_cost = random.randint(min_r_cost,max_r_cost)
        print(min_r_cost, max_r_cost, stopping_r_cost)
        
        log = list()
        
        r_i_cost = 0
        r_i_finished = False
        r_id = -1
        v_id = None
        for i in range(len(seq)-1):
            if seq[i] == 0:
                r_id += 1
                v_id = self.route_to_vehicle[r_id]
                r_i_cost = 0
                r_i_finished = False
            elif r_i_finished is False:
                if r_i_cost + self.tasks[seq[i]].serv_cost <= stopping_r_cost:
                    log.append((v_id, (self.tasks[seq[i]].head_node, self.tasks[seq[i]].tail_node), 1))
                    r_i_cost += self.tasks[seq[i]].serv_cost
                else:
                    # print(r_i_cost)
                    r_i_finished = True
            if r_i_finished is False:
                for j in range(1,self.shortest_path[self.tasks[seq[i]].tail_node][self.tasks[seq[i+1]].head_node][0]):
                    head_node = self.shortest_path[self.tasks[seq[i]].tail_node][self.tasks[seq[i+1]].head_node][j]
                    tail_node = self.shortest_path[self.tasks[seq[i]].tail_node][self.tasks[seq[i+1]].head_node][j+1]
                    if r_i_cost + self.trav_cost[head_node][tail_node] <= stopping_r_cost:
                        log.append((v_id, (head_node, tail_node), 0))
                        r_i_cost += self.trav_cost[head_node][tail_node]
                    else:
                        # print(r_i_cost)
                        r_i_finished = True
                        break
        
        return log
    
    
    def generate_task_appearance_event(self, log):
        served_arcs = self.served_arcs_from_log(log)
        required_arcs = set([(arc.head_node, arc.tail_node) for arc in self.tasks])
        all_arcs = set([(arc.head_node, arc.tail_node) for arc in self.arcs])
        # unrequired arcs/edges and required arcs/edges that have been already served (based on the log)
        potential_arcs = all_arcs.difference(required_arcs).union(served_arcs)
        
        if len(potential_arcs) == 0:
            return None, None, None
        
        # choose an arc
        new_task_arc = random.sample(potential_arcs, k=1)[0]
        
        # generate demand
        new_task_demand = random.randint(1,int(self.capacity/3))
        
        # get serv_cost (= trav_cost)
        new_task_arc_id = self.find_arc_id(new_task_arc[0], new_task_arc[1])
        new_task_serv_cost = self.arcs[new_task_arc_id].trav_cost        
        
        print(new_task_arc, new_task_demand, new_task_serv_cost)
        
        return new_task_arc, new_task_demand, new_task_serv_cost
        
    
    def generate_demand_increased_event(self, log):
        served_arcs = self.served_arcs_from_log(log)
        required_arcs = set([(arc.head_node, arc.tail_node) for arc in self.tasks])
        unserved_arcs = required_arcs.difference(served_arcs)
        unserved_arcs.remove((1,1))
      
        if len(unserved_arcs) == 0:
            return None, None, None
        
        # choose an arc
        di_task_arc = random.sample(unserved_arcs, k=1)[0]
        
        # choose the percentage of the increase for demand and service cost
        di_task_arc_id = self.find_task_id(di_task_arc[0], di_task_arc[1])
        curr_demand = self.tasks[di_task_arc_id].demand     
        incr_percentage = random.randint(1,100)/100
        while curr_demand + incr_percentage * curr_demand > self.capacity or \
            int(incr_percentage * curr_demand) <= 0:
            incr_percentage = random.randint(1,100)/100
        
        # calculate the increase in demand
        di_task_demand_increase = int(curr_demand * incr_percentage)
        
        # calculate the increase in service cost
        di_task_serv_cost_increase = int(self.tasks[di_task_arc_id].serv_cost * incr_percentage)
        
        print(di_task_arc_id, di_task_arc, di_task_demand_increase, di_task_serv_cost_increase)
        
        return di_task_arc, di_task_demand_increase, di_task_serv_cost_increase
    
    
    def generate_vehicle_breakdown_event(self, log):
        outside_vehicles = set()
        for event in log:
            if event[1][0] == self.depot:
                outside_vehicles.add(event[0])
            elif event[1][1] == self.depot:
                outside_vehicles.remove(event[0])
        
        if len(outside_vehicles) == 0:
            return None
        
        broken_v_id = random.sample(outside_vehicles, k=1)[0]
        
        print(broken_v_id)
        
        return broken_v_id
        
        
    def served_arcs_from_log(self, log):
        served_arcs = set()
        for event in log[self.last_processed_event_id+1:len(log)]:
            if event[2] == 1:
                task_id = self.find_task_id(event[1][0], event[1][1])
                if task_id != -1:
                    served_arcs.add((event[1][0], event[1][1]))
                    if self.tasks[task_id].inverse != None:
                        served_arcs.add((event[1][1], event[1][0]))                 
                else:
                    print("Error: unknown task in the log on arc "+str(event))
        return served_arcs
    
    
    # collect the IDs of the served tasks 
    def served_task_ids_from_log(self, log):
        served_task_ids = set()        
        for event in log[self.last_processed_event_id+1:len(log)]:
            if event[2] == 1:
                # find the ID of a task by head and tail node
                task_id = self.find_task_id(event[1][0], event[1][1])
                if task_id != -1:
                    served_task_ids.add(task_id)
                    if self.tasks[task_id].inverse != None:
                        served_task_ids.add(self.tasks[task_id].inverse) 
                else:
                    print("Error: unknown task in the log on arc "+str(event))
        return served_task_ids
    
    
    def create_virtual_tasks(self, ind, log, exception_v_id=-1):
        seq = list()
        exception_v_task_ids = list()
        # create virtual tasks and new individual (solution)
        for r_id in range(len(ind.route_seg)):
            v_id = self.route_to_vehicle[r_id]
            v_last_node = None
            # calculate the total cost and served demand
            cost = 0
            demand = 0
            for e_id in range(self.last_processed_event_id+1, len(log)):               
                if log[e_id][0] == v_id:
                    v_last_node = log[e_id][1][1]
                    # if reached the depot
                    if log[e_id][1][1] == self.depot:
                        self.finished_routes.add(r_id)
                        # if there are no more routes assigned to it and there are unassigned routes
                        if v_id not in self.route_to_vehicle[r_id+1:] and \
                            None in self.route_to_vehicle:
                                r_id = min(r_id for r_id, _v_id in enumerate(self.route_to_vehicle) if _v_id == None)
                                self.route_to_vehicle[r_id] = v_id
                        else:
                            self.free_vehicles.add(v_id)
                    # served arc -> find task (arc)
                    if log[e_id][2] == 1:
                        task_id = self.find_task_id(log[e_id][1][0], log[e_id][1][1])
                        if task_id != -1:
                            demand += self.tasks[task_id].demand
                            cost += self.tasks[task_id].serv_cost
                        else:
                            print("Error: unknown task in the log on arc "+str(log[e_id]))
                    # traversed arc -> find arc
                    elif log[e_id][2] == 0:
                        arc_id = self.find_arc_id(log[e_id][1][0], log[e_id][1][1])
                        if arc_id != -1:
                            cost += self.arcs[arc_id].trav_cost
                        else:
                            print("Error: unknown arc in the log: "+str(log[e_id]))
            
            # if it is a new route (i.e., doesn't have vt yet) and has been started to be executed
            if self.virtual_task_ids[r_id] == 0 and v_last_node != None:
                virtual_task = Task(self.depot, v_last_node, inf, cost, demand, None)
                virtual_task.id = len(self.tasks)
                self.tasks.append(virtual_task)
                vt_id = len(self.tasks)-1
                self.virtual_task_ids[r_id] = vt_id
                self.task_ids.add(vt_id) 
                self.task_ids_with_inv.add(vt_id)
            # if it has a vt and moved since last examination
            elif v_last_node != None:
                self.tasks[self.virtual_task_ids[r_id]].tail_node = v_last_node
                self.tasks[self.virtual_task_ids[r_id]].demand += demand
                self.tasks[self.virtual_task_ids[r_id]].serv_cost += cost

            seq.append(0)
            
            if self.virtual_task_ids[r_id] != 0:
                seq.append(vt_id)
            
            if v_id != exception_v_id:
                for task_id in ind.sequence[ind.route_seg[r_id][0]+1:ind.route_seg[r_id][1]]:
                    if task_id in self.task_ids_with_inv:
                        seq.append(task_id)
            else:
                for task_id in ind.sequence[ind.route_seg[r_id][0]+1:ind.route_seg[r_id][1]]:
                    if task_id in self.task_ids_with_inv:
                        exception_v_task_ids.append(task_id)
            
        seq.append(0)
        
        self.last_processed_event_id = len(log)-1
        
        if exception_v_id != -1:
            
            seq2 = copy.deepcopy(seq)
            if len(exception_v_task_ids) != 0:
                for task_id in exception_v_task_ids:
                    seq2.append(task_id)
                seq2.append(0)
            
            #print(seq, seq2, exception_v_task_ids)
            return seq, seq2, exception_v_task_ids
        
        return seq
    
    
    # nt_type: edge (0) / arc (1)
    def event_new_task(self, ind, log, nt_arc, nt_serv_cost, nt_demand, nt_type):

        if nt_type not in {0, 1}:
            print("Error: invalid task type")
            return None, None
        if nt_demand > self.capacity:
            print("Error: the demand of the new task is greater than the maximum capacity of the vehicles")
            return None, None
        
        # get the arc(s) of the task(s)
        arc_id = self.find_arc_id(nt_arc[0], nt_arc[1])
        inv_arc_id = self.find_arc_id(nt_arc[1], nt_arc[0]) if nt_type == 0 else None
       
        if (nt_type == 0 and (arc_id == -1 or inv_arc_id == -1)) or (nt_type == 1 and arc_id == -1):
            print("Error: the new task cannot be added to an unknown arc")
            return None, None
        
        new_instance = copy.deepcopy(self)
        new_task_id = len(self.tasks)
        new_inv_task_id = len(self.tasks)+1 if nt_type == 0 else None
        
        # create and add new task(s)
        if nt_type == 0:
            new_task = Task(self.arcs[arc_id].head_node, self.arcs[arc_id].tail_node, self.arcs[arc_id].trav_cost, \
                        nt_serv_cost, nt_demand, new_inv_task_id)
            new_inv_task = Task(self.arcs[inv_arc_id].head_node, self.arcs[inv_arc_id].tail_node, self.arcs[inv_arc_id].trav_cost, \
                        nt_serv_cost, nt_demand, new_task_id)
            new_instance.tasks.append(new_task)
            new_instance.tasks.append(new_inv_task)
            new_instance.task_ids.add(new_task_id)
            new_instance.task_ids_with_inv.update({new_task_id, new_inv_task_id})
        elif nt_type == 1:
            new_task = Task(self.arcs[arc_id].head_node, self.arcs[arc_id].tail_node, self.arcs[arc_id].trav_cost, \
                        nt_serv_cost, nt_demand, None)
            new_instance.tasks.append(new_task)
            new_instance.task_ids.add(new_task_id)
            new_instance.task_ids_with_inv.add(new_task_id)
        
        # remove the IDs of the served tasks (inv. too) from the IDs of the unserved tasks
        served_task_ids = self.served_task_ids_from_log(log)
        new_instance.task_ids_with_inv.difference_update(served_task_ids)
        new_instance.task_ids.difference_update(served_task_ids)
        
        # create virtual tasks and new sequence
        ind_wo_nr_seq = new_instance.create_virtual_tasks(ind, log)
        ind_w_nr_seq = copy.deepcopy(ind_wo_nr_seq) + [new_task_id, 0]
        #print(ind_w_nr_seq, ind_wo_nr_seq, new_task_id)

        return new_instance, (ind_w_nr_seq, ind_wo_nr_seq, [new_task_id])

    # t_type: edge (0) / arc (1)
    def event_increased_demand(self, ind, log, t_arc, t_serv_cost_incr, t_demand_incr, t_type):
        
        if t_type not in {0, 1}:
            print("Error: invalid task type")
            return None, None, (None, None, None)

        served_task_ids = self.served_task_ids_from_log(log)
        
        # get the task(s)
        task_id = self.find_task_id(t_arc[0], t_arc[1])
        inv_task_id = self.tasks[task_id].inverse if t_type == 0 else None
        print("The task id(s):", task_id, inv_task_id)
        print("The old and the new demand:", self.tasks[task_id].demand, t_demand_incr + self.tasks[task_id].demand)

        if (t_type == 0 and (task_id == -1 or inv_task_id == -1)) or (t_type == 1 and task_id == -1):
            print("Error: the demand of an unknow task cannot be increased")
            return None, None, (None, None, None)
        if task_id in served_task_ids:
            print("Error: the demand of an already served task cannot be increased")
            return None, None, (None, None, None)
        if t_demand_incr + self.tasks[task_id].demand > self.capacity:
            print("Error: the new demand of the task is greater than the maximum capacity of the vehicles")
            return None, None, (None, None, None)
        
        # update the demand of the task(s)
        new_instance = copy.deepcopy(self)
        new_instance.tasks[task_id].demand += t_demand_incr
        new_instance.tasks[task_id].serv_cost += t_serv_cost_incr
        if t_type == 0:
            new_instance.tasks[inv_task_id].demand += t_demand_incr
            new_instance.tasks[inv_task_id].serv_cost += t_serv_cost_incr      
        
        # check if the vehicle still can serve all the tasks on its planned route
        v_demand = t_demand_incr
        r_id = -1
        v_id = -1
        # find v_id
        for t_id in ind.sequence:
            if t_id == 0:
                r_id += 1
            if t_id == task_id or t_id == inv_task_id:
                v_id = self.route_to_vehicle[r_id]
                break
        # calculate the total demand of the served tasks
        for event in log:
            if event[0] == v_id and event[2] == 1:
                t_id = self.find_task_id(event[1][0], event[1][1])
                if t_id != -1:
                    v_demand += self.tasks[t_id].demand
                else:
                    print("Error: unknown task in the log on arc "+str(event))
        # calculate the total demand of the planned tasks (includind the tasl in question)
        current_route_plan = ind.sequence[ind.route_seg[r_id][0]+1:ind.route_seg[r_id][1]]
        for t_id in current_route_plan:
            if t_id not in served_task_ids:
                v_demand += self.tasks[t_id].demand
        
        # if the capacity constraint still holds -> no rerouting is needed
        excess_demand =  v_demand - self.capacity
        if excess_demand <= 0:
            print("No rerouting is needed.")
            return new_instance, 0, (ind, None, None)
        else:
            print("The demand exceeds the capacity by", excess_demand,". Rerouting is needed.")
            
        # remove the IDs of the served tasks (inv. too) from the IDs of the unserved tasks
        new_instance.task_ids_with_inv.difference_update(served_task_ids)
        new_instance.task_ids.difference_update(served_task_ids)
        
        # create virtusl tasks and new sequence
        ind_wo_nr_seq = new_instance.create_virtual_tasks(ind, log)              

        if task_id in ind_wo_nr_seq:
            ind_wo_nr_seq.remove(task_id)
        elif inv_task_id in ind_wo_nr_seq:
            ind_wo_nr_seq.remove(inv_task_id)
        
        ind_w_nr_seq = copy.deepcopy(ind_wo_nr_seq) + [task_id, 0]
        #print(ind_w_nr_seq, ind_wo_nr_seq, task_id)
        
        return new_instance, 1, (ind_w_nr_seq, ind_wo_nr_seq, [task_id])
        

    # log: the set of event (travel and service) logs of the vehicles
    #   - log[v][i][0]: vehicle v, i-th event, traversed arc
    #   - log[v][i][1]: vehicle v, i-th event, only traversed (0) / served (1)
    # broken_v_id: the ID of the broken down vehicle
    def event_vehicle_breakdown(self, ind, log, broken_v_id):
        
        new_instance = copy.deepcopy(self)

        # remove the IDs of the served tasks (inv. too) from the IDs of the unserved tasks
        served_task_ids = self.served_task_ids_from_log(log)
        new_instance.task_ids_with_inv.difference_update(served_task_ids)
        new_instance.task_ids.difference_update(served_task_ids)

        # create virtual tasks and new sequence
        ind_wo_nr_seq, ind_w_nr_seq, bv_task_ids = new_instance.create_virtual_tasks(ind, log, broken_v_id)
        #print(ind_w_nr_seq, ind_wo_nr_seq, bv_task_ids)
        
#        new_instance.broken_vehicles.add(broken_v_id)
        r_ids = [r_id for r_id, v_id in enumerate(self.route_to_vehicle) if v_id == broken_v_id]
        current_r_id = max(r_ids)
        new_instance.finished_routes.add(current_r_id)
        
        print("The ID of the broken vehicle:", broken_v_id)
        print("The ID of the affected route plan:", current_r_id)
        
        # check whether the broken down vehicle still have tasks to serve
        if len(bv_task_ids) == 0:
            print("No rerouting is needed.")
            return new_instance, 0, (new_instance.ind_from_seq(ind_wo_nr_seq), None, None)
        else:
            print("Rerouting is needed. The affected task(s):")
            for task_id in bv_task_ids:
                print(task_id, self.tasks[task_id].inverse)
            return new_instance, 1, (ind_w_nr_seq, ind_wo_nr_seq, bv_task_ids)
    

# -------------------------------------- RR1 ------------------------------------------------

    
    def reroute_one_route(self, ind_w_nr_seq, ind_wo_nr_seq, task_ids):
        start_time = timer()
        
        # create individual with new route which contains the tasks of the broken down vehicle
        ind_w_nr = self.ind_from_seq(ind_w_nr_seq)       
        # create individual without the tasks of the broken down vehicle
        ind_wo_nr = self.ind_from_seq(ind_wo_nr_seq)

        tasks_demand = 0
        for task_id in task_ids:
            tasks_demand += self.tasks[task_id].demand
        
        potential_route_ids = set()
        for r_id in range(len(ind_wo_nr.route_seg)):
            if r_id not in self.finished_routes and \
                tasks_demand + ind_wo_nr.route_seg_load[r_id] <= self.capacity:
                potential_route_ids.add(r_id)
        
        if len(potential_route_ids) == 0:
            self.virtual_task_ids.append(0)
            if len(self.free_vehicles) != 0:
                v_id = random.sample(self.free_vehicles, k=1)[0]
                self.free_vehicles.remove(v_id)
            else:
                v_id = None
            self.route_to_vehicle.append(v_id)
            return ind_w_nr
               
        best_ind = ind_w_nr 
        for r_id in potential_route_ids:
            ind = copy.deepcopy(ind_wo_nr)
            for task_id in task_ids:
                r_task_best_i_cost_diff = inf
                r_task_best_i = -1
                r_task_best_i_inv = False
                for i in range(ind.route_seg[r_id][0]+2,ind.route_seg[r_id][1]+1):
                    
                    cost_diff = self.tasks[task_id].serv_cost \
                    + self.min_cost[self.tasks[ind.sequence[i-1]].tail_node][self.tasks[task_id].head_node] \
                    + self.min_cost[self.tasks[task_id].tail_node][self.tasks[ind.sequence[i]].head_node] \
                    - self.min_cost[self.tasks[ind.sequence[i-1]].tail_node][self.tasks[ind.sequence[i]].head_node]
                    
                    cost_diff_inv = inf if self.tasks[task_id].inverse == None \
                    else self.tasks[self.tasks[task_id].inverse].serv_cost \
                    + self.min_cost[self.tasks[ind.sequence[i-1]].tail_node][self.tasks[task_id].tail_node] \
                    + self.min_cost[self.tasks[task_id].head_node][self.tasks[ind.sequence[i]].head_node] \
                    - self.min_cost[self.tasks[ind.sequence[i-1]].tail_node][self.tasks[ind.sequence[i]].head_node]
                    
                    if cost_diff < r_task_best_i_cost_diff:
                        r_task_best_i_cost_diff = cost_diff
                        r_task_best_i = i
                        r_task_best_i_inv = False
                    elif cost_diff_inv < r_task_best_i_cost_diff:
                        r_task_best_i_cost_diff = cost_diff_inv
                        r_task_best_i = i
                        r_task_best_i_inv = True
                
                t_id = self.tasks[task_id].inverse if r_task_best_i_inv else task_id
                ind.sequence.insert(r_task_best_i, t_id)
                ind.route_seg[r_id] = (ind.route_seg[r_id][0], ind.route_seg[r_id][1]+1)
                if r_id < len(ind.route_seg)-1:
                    for r_id_right in range(r_id+1, len(ind.route_seg)):
                        ind.route_seg[r_id_right] = (ind.route_seg[r_id_right][0]+1, ind.route_seg[r_id_right][1]+1)
                ind.route_seg_load[r_id] += self.tasks[t_id].demand
                ind.total_cost += r_task_best_i_cost_diff
            
            # if ind cost less or the same but have less routes than the current best ind -> new best ind
            if ind.total_cost < best_ind.total_cost:
                best_ind = ind
        
        if best_ind == ind_w_nr:
            self.virtual_task_ids.append(0)
            if len(self.free_vehicles) != 0:
                v_id = random.sample(self.free_vehicles, k=1)[0]
                self.free_vehicles.remove(v_id)     
            else:
                v_id = None
            self.route_to_vehicle.append(v_id)
        
        print(timer()-start_time)
        
        return best_ind
        

# --------------------------------- ABC --------------------------------------------

    def abc(self, colony_size=10, max_number=100000, no_improvement_limit=200, \
            local_search_limit=20, max_solution_age=10, initial_solution=None):
        start_time = timer()
        self.abc_initialize_population(colony_size, initial_solution)
        self.best_solution = min(self.population, key=attrgetter('total_cost'))
        current_best = copy.deepcopy(self.best_solution)
        c = 0
        no_improvement = 0
        while no_improvement < no_improvement_limit and timer()-start_time <= 600:
            self.employed_bee_phase(local_search_limit)
            self.onlooker_bee_phase(local_search_limit)
            self.scout_bee_phase(max_solution_age)

            if current_best.total_cost == self.best_solution.total_cost:
                no_improvement += 1
            else:
                current_best = copy.deepcopy(self.best_solution)
                real_solution = self.add_finished_routes(copy.deepcopy(current_best.sequence))
                print(str(real_solution.total_cost)+"\t"+str(timer()-start_time)+"\t"+str(c))
                no_improvement = 0
            
            c += 1
            
        real_solution = self.add_finished_routes(self.best_solution.sequence)
        return real_solution
    

    def abc_initialize_population(self, pop_size, initial_solution=None):        
        self.population_size = pop_size
        self.population = list(np.zeros(pop_size, dtype=int))
        self.population_probability = np.zeros(pop_size, dtype=float)

        start_id = 0
        if initial_solution != None:
            self.population[0] = initial_solution
            self.population[0].index = 0
            start_id = 1
        
        for i in range(start_id, pop_size):
            self.population[i] = self.random_routing_plan_generator(self.task_ids)
            self.population[i].index = i
        
        
        """
        sols = list()
        for i in range(100):
            sols.append(self.random_routing_plan_generator(self.task_ids))
        sols.sort(key=lambda x: x.total_cost, reverse=False)
        for i in range(pop_size):
            ind = sols[i]
            ind.index = i
            self.population[i] = ind
        """
        
        self.best_solution = min(self.population, key=attrgetter('total_cost'))
    
    
    def employed_bee_phase(self, search_trial):
        lista = [0] * 6
        lista2 = [0] * 6
        lista3 = [0] * 6
        for i, ind in enumerate(self.population):  
            best_ind_i = copy.deepcopy(ind)
            best_ind_i.age = 0
            while best_ind_i.age < search_trial:
                new_inds = [self.inverse(best_ind_i), \
                            self.insert(best_ind_i), \
                            self.swap(best_ind_i), \
                            self.two_opt(best_ind_i), \
                            self.greedy_sub_tour_mutation(best_ind_i)]
                best_ind = min([best_ind_i] + new_inds, key=attrgetter('total_cost'))
                
                #print([x.total_cost for x in new_inds])
                #print(str(best_ind.total_cost)+"\n")
                op_idx = new_inds.index(best_ind) if best_ind in new_inds else 5
                lista[op_idx] += 1
                
                if best_ind.total_cost <= best_ind_i.total_cost and \
                    best_ind.sequence != best_ind_i.sequence:
                    lista2[op_idx] += 1
                    best_ind_i = best_ind
                    best_ind_i.age = 0
                    if best_ind_i.total_cost < self.best_solution.total_cost:
                        self.best_solution = copy.deepcopy(best_ind_i)
                        lista3[op_idx] += 1
                else:
                    best_ind_i.age += 1
            if best_ind_i.sequence == ind.sequence:
                self.population[i].age += 1
            else:
                self.population[i] = best_ind_i
                self.population[i].age = 0
                self.population[i].index = i
        print(lista, lista2, lista3)
        
        """
        total_fitness = sum(ind.fitness for ind in self.population)
        for i, ind in enumerate(self.population):
            self.population_probability[i] = self.population[i].fitness / total_fitness
        #print(self.population_probability)
        """

    def onlooker_bee_phase(self, search_trial):
        #best_inds = random.choices(self.population, weights = self.population_probability, k = 3)
        #ind = max(best_inds, key=attrgetter('fitness'))
        best_inds = random.choices(self.population, k = 3)
        ind = min(best_inds, key=attrgetter('total_cost'))

        neighbour_inds = [self.merge_split(ind) for i in range(self.population_size)]
        for i in range(self.population_size):
            neighbour_inds[i].age = 0
            best_ind_i = copy.deepcopy(neighbour_inds[i])
            best_ind_i.age = 0
            next_ind_i = copy.deepcopy(best_ind_i)
            while best_ind_i.age < search_trial:
                new_inds = [self.inverse(next_ind_i), \
                            self.insert(next_ind_i, only_feasible=False), \
                            self.swap(next_ind_i, only_feasible=False), \
                            self.two_opt(next_ind_i, only_feasible=False), \
                            self.greedy_sub_tour_mutation(next_ind_i)]
                #next_ind_i = min([next_ind_i] + new_inds, key=attrgetter('total_cost'))
                next_ind_i = min(new_inds, key=attrgetter('total_cost'))
                if next_ind_i.total_cost <= best_ind_i.total_cost and \
                    next_ind_i.sequence != best_ind_i.sequence and \
                    self.calculate_excess_demand(next_ind_i.route_seg_load)==0:
                    best_ind_i = next_ind_i
                    best_ind_i.age = 0
                else:
                    best_ind_i.age += 1
            if best_ind_i.sequence == neighbour_inds[i].sequence:
                neighbour_inds[i].age += 1
            else:
                neighbour_inds[i] = best_ind_i
                neighbour_inds[i].age = 0
            neighbour_inds[i].index = ind.index 
            #print(neighbour_inds[i].total_cost, neighbour_inds[i].age)
            
        best_neighbour_ind = min(neighbour_inds + [ind], key=attrgetter('total_cost'))
        #print(best_neighbour_ind.total_cost, ind.total_cost, self.population[ind.index].total_cost)
        
        if best_neighbour_ind.total_cost < ind.total_cost:
            self.population[ind.index] = best_neighbour_ind
            self.population[ind.index].age = 0
        else:
            self.population[ind.index].age += 1
        
        self.best_solution = min(self.population + [self.best_solution], key=attrgetter('total_cost'))
        
    
    def scout_bee_phase(self, x):
        for i, ind in enumerate(self.population):
            if ind.age > x:                
                self.population[i] = self.random_routing_plan_generator(self.task_ids)
                print("new solution")
    
    
    def print_population_total_cost(self):
        return list((ind.total_cost, ind.age) for ind in self.population)
    

# --------------------------------- HMA -----------------------------------------

    # psize: population size
    # r: a set of treshold ratios
    # w: the number of non-improving attractors visited
    def hma(self, psize=10, r=[0.003, 0.004, 0.005, 0.006], w=5):
        self.start_time = timer()
        self.time_limit = 600
        # population initialization
        self.hma_initialize_population(psize)
        # record best solution so far
        self.best_solution = min(self.population, key=attrgetter('total_cost'))
        real_solution = self.add_finished_routes(copy.deepcopy(self.best_solution.sequence))
        print(str(real_solution.total_cost)+"\t"+str(timer()-self.start_time)+"\t0")
        # initialize the counter array
        cnt = list(np.ones(len(r), dtype=int))
        # main search procedure
        x = 0
        while x<1 and timer()-self.start_time <= self.time_limit:
            #print(str(x)+". iteration")
            # randomly select 2 solutions
            ind1, ind2 = random.sample(self.population, 2)
            #print(ind1.sequence, ind1.total_cost)
            #print(ind2.sequence, ind2.total_cost)
            # Route-based crossover
            ind0 = self.route_based_crossover(ind1, ind2)
            #print("Route-based crossover")
            #print(ind0.sequence, ind0.total_cost)
            # Determine the order of conducting RTTP (0) and IDP (1)
            od = random.randint(0,1)
            # Select a ratio
            k = self.probabilistic_select_ratio(cnt)
            # Improve the solution
            ind0 = self.local_refinement(ind0, r[k], w, od)
            # Pool updating
            new_solution = False
            if self.calculate_quality_and_distance_fitness(ind0):
                new_best_solution = min(self.population, key=attrgetter('total_cost'))
                if new_best_solution.total_cost < self.best_solution.total_cost:
                    new_solution = True
                self.best_solution = new_best_solution
                cnt[k] += 1
            #print("Global best")
            real_solution = self.add_finished_routes(copy.deepcopy(self.best_solution.sequence))
            if new_solution:
                print(str(real_solution.total_cost)+"\t"+str(timer()-self.start_time)+"\t"+str(x))
            #print()
            x += 1
        return real_solution
    
    
    def hma_initialize_population(self, pop_size):        
        self.population_size = pop_size       
        self.population = list(np.zeros(pop_size, dtype=int))
    
        self.population[0] = self.randomized_path_scanning_heuristic(self.task_ids_with_inv)       
        for i in range(1,pop_size):
            ind = self.random_routing_plan_generator(self.task_ids)
            ind.index = i
            self.population[i] = ind

    
    def probabilistic_select_ratio(self, cnt):
        pr = copy.deepcopy(cnt)
        sum_cnt = sum(cnt)
        cnt_indexes = list(range(len(cnt)))
        for i in cnt_indexes:
            pr[i] = cnt[i] / sum_cnt
        return random.choices(cnt_indexes, weights = pr, k = 1)[0]

    
    def local_refinement(self, ind0, r, w, od):
        # RTTP first
        if od == 0:
            ind0 = self.rttp(r, w, ind0)
            ind0 = self.idp(ind0)
        # IDP first
        else:
            ind0 = self.idp(ind0)
            ind0 = self.rttp(r, w, ind0)
        return ind0
    

    # tasks1 - tasks2 (inverse tasks are taken into account too!)
    def route_difference(self, tasks1, tasks2):
        tasks2_w_inv = copy.deepcopy(tasks2)                                  
        for task in tasks2:
            if self.tasks[task].inverse != None:
                tasks2_w_inv.add(self.tasks[task].inverse)
        return tasks1.difference(tasks2_w_inv)
    
    
    def route_based_crossover(self, ind1, ind2):
        ind0_seq = copy.deepcopy(ind1.sequence)
        a = random.randint(0,len(ind1.route_seg)-1)
        b = random.randint(0,len(ind2.route_seg)-1)
        ind0_seq[ind1.route_seg[a][0]:ind1.route_seg[a][1]+1] = \
            copy.deepcopy(ind2.sequence[ind2.route_seg[b][0]:ind2.route_seg[b][1]+1])
        
        ind1_ra_tasks = set(ind1.sequence[ind1.route_seg[a][0]+1:ind1.route_seg[a][1]])
        ind2_rb_tasks = set(ind2.sequence[ind2.route_seg[b][0]+1:ind2.route_seg[b][1]])
        
        unserved_tasks = self.route_difference(ind1_ra_tasks, ind2_rb_tasks)
        duplicate_tasks = self.route_difference(ind2_rb_tasks, ind1_ra_tasks)
        
        """
        print(ind1.sequence)
        print(ind2.sequence)
        print(a, b)
        print(unserved_tasks)
        print(duplicate_tasks)
        print(ind0_seq)
        """
        
        # remove the duplicate tasks from the positions which results with a better total cost
        for task in duplicate_tasks:
            task_pos = [i for i, t in enumerate(ind0_seq) if t == task or t == self.tasks[task].inverse]
            
            # cost of keeping task at pos0 and removing from pos1
            keeping_task_at_pos0_cost = \
                self.min_cost[self.tasks[ind0_seq[task_pos[0]-1]].tail_node, \
                              self.tasks[ind0_seq[task_pos[0]]].head_node] \
                + self.min_cost[self.tasks[ind0_seq[task_pos[0]]].tail_node, \
                                self.tasks[ind0_seq[task_pos[0]+1]].head_node] \
                + self.min_cost[self.tasks[ind0_seq[task_pos[1]-1]].tail_node,\
                                self.tasks[ind0_seq[task_pos[1]+1]].head_node]
            
            # cost of keeping task at pos1 and removing from pos0
            keeping_task_at_pos1_cost = \
                self.min_cost[self.tasks[ind0_seq[task_pos[1]-1]].tail_node, \
                              self.tasks[ind0_seq[task_pos[1]]].head_node] \
                + self.min_cost[self.tasks[ind0_seq[task_pos[1]]].tail_node, \
                                self.tasks[ind0_seq[task_pos[1]+1]].head_node] \
                + self.min_cost[self.tasks[ind0_seq[task_pos[0]-1]].tail_node,\
                                self.tasks[ind0_seq[task_pos[0]+1]].head_node]
            
            if keeping_task_at_pos0_cost < keeping_task_at_pos1_cost:
                del(ind0_seq[task_pos[1]])
            else:
                del(ind0_seq[task_pos[0]])
        
        # remove excess 0s (it is needed if a route plan became empty)
        task_pos = 1
        while task_pos < len(ind0_seq):
            if ind0_seq[task_pos-1] == 0 and ind0_seq[task_pos] == 0:
                del(ind0_seq[task_pos])
            else:
                task_pos += 1
        if ind0_seq[-1] != 0:
            ind0_seq.append(0)
        
        # sort the unserved tasks in random order
        duplicate_tasks = list(duplicate_tasks) 
        random.shuffle(duplicate_tasks)
        
        #print(ind0_seq)
        
        ind0 = self.ind_from_seq(ind0_seq)
        
        for task_id in unserved_tasks:
            potential_route_ids = set()
            for r_id in range(len(ind0.route_seg)):
                if ind0.route_seg_load[r_id] + self.tasks[task_id].demand <= self.capacity:
                    potential_route_ids.add(r_id)
            
            if potential_route_ids != set():       
                r_task_best_i_cost_diff = inf   # the cost difference of the best position
                r_task_best_r = -1              # the route plan id of the best position
                r_task_best_i = -1              # the best position
                r_task_best_i_inv = False
                for r_id in potential_route_ids:
                    for i in range(ind0.route_seg[r_id][0]+2,ind0.route_seg[r_id][1]+1):                  
                        cost_diff = self.tasks[task_id].serv_cost \
                        + self.min_cost[self.tasks[ind0.sequence[i-1]].tail_node][self.tasks[task_id].head_node] \
                        + self.min_cost[self.tasks[task_id].tail_node][self.tasks[ind0.sequence[i]].head_node] \
                        - self.min_cost[self.tasks[ind0.sequence[i-1]].tail_node][self.tasks[ind0.sequence[i]].head_node]
                        
                        cost_diff_inv = inf if self.tasks[task_id].inverse == None \
                        else self.tasks[self.tasks[task_id].inverse].serv_cost \
                        + self.min_cost[self.tasks[ind0.sequence[i-1]].tail_node][self.tasks[task_id].tail_node] \
                        + self.min_cost[self.tasks[task_id].head_node][self.tasks[ind0.sequence[i]].head_node] \
                        - self.min_cost[self.tasks[ind0.sequence[i-1]].tail_node][self.tasks[ind0.sequence[i]].head_node]
                        
                        if cost_diff < r_task_best_i_cost_diff:
                            r_task_best_i_cost_diff = cost_diff
                            r_task_best_r = r_id
                            r_task_best_i = i
                            r_task_best_i_inv = False
                        elif cost_diff_inv < r_task_best_i_cost_diff:
                            r_task_best_i_cost_diff = cost_diff_inv
                            r_task_best_r = r_id
                            r_task_best_i = i
                            r_task_best_i_inv = True
                    
                t_id = self.tasks[task_id].inverse if r_task_best_i_inv else task_id
                ind0.sequence.insert(r_task_best_i, t_id)
                ind0.route_seg[r_task_best_r] = (ind0.route_seg[r_task_best_r][0], ind0.route_seg[r_task_best_r][1]+1)
                if r_task_best_r < len(ind0.route_seg)-1:
                    for r_id_right in range(r_task_best_r+1, len(ind0.route_seg)):
                        ind0.route_seg[r_id_right] = (ind0.route_seg[r_id_right][0]+1, ind0.route_seg[r_id_right][1]+1)
                ind0.route_seg_load[r_task_best_r] += self.tasks[t_id].demand
                ind0.total_cost += r_task_best_i_cost_diff
            else:
                ind0.sequence.extend([task_id, 0])
                ind0 = self.ind_from_seq(ind0.sequence)
        
        #print(ind0.sequence)
        
        return ind0
    
    
    # Randomized Tabu Tresholding Procedure (RTTP)
    # r: threshold ratio
    # W: the number of non-improving attractors visited
    # ind: an initial solution
    def rttp(self, r, W, ind):
        operators = [self.inverse, self.insert, self.double_insert, self.swap, self.two_opt]
        ind_global_best = ind
        ind_curr = ind
        ind_local_best_total_cost = ind_global_best.total_cost
        w = 0
        while w < W:
            # Tabu timing value
            T = 5 # random.randint(28,33)
            
            # Mixed phase
            #print("RTTP Mixed phase")
            for k in range(1,T+1):
                random.shuffle(operators)
                for operator in operators:
                    task_ids = self.get_task_ids_from_sequence(ind_curr.sequence)
                    random.shuffle(task_ids)
                    for task_id in task_ids:
                        if operator != self.inverse:
                            best_feasible_move_ind = ind
                            task_id_candidate_list = self.construct_candidate_list(task_id, task_ids)
                            for task_id_2 in task_id_candidate_list:
                                task_id = task_id if task_id in ind_curr.sequence else self.tasks[task_id].inverse
                                task_id_2 = task_id_2 if task_id_2 in ind_curr.sequence else self.tasks[task_id_2].inverse                       
                                new_ind = operator(ind_curr, task_id, task_id_2)
                                if new_ind.total_cost <= (1 + r) * ind_local_best_total_cost and \
                                    new_ind.total_cost < best_feasible_move_ind.total_cost:
                                    best_feasible_move_ind = new_ind
                                    if new_ind.total_cost < ind_global_best.total_cost:
                                        ind_global_best = new_ind
                                        # temp
                                        if ind_global_best.total_cost < self.best_solution.total_cost and timer()-self.start_time <= self.time_limit:
                                            print(str(ind_global_best.total_cost)+"\t"+str(timer()-self.start_time)+"\t0")
                                        break
                            ind_curr = best_feasible_move_ind
                        else:
                            new_ind = operator(ind_curr, task_id)
                            if new_ind.total_cost <= (1 + r) * ind_local_best_total_cost:
                                ind_curr = new_ind
                                if new_ind.total_cost < ind_global_best.total_cost:
                                    ind_global_best = new_ind
                                    # temporaly added
                                    if ind_global_best.total_cost < self.best_solution.total_cost and timer()-self.start_time <= self.time_limit:
                                        print(str(ind_global_best.total_cost)+"\t"+str(timer()-self.start_time)+"\t0")
            
            #print(ind_global_best.sequence, ind_global_best.route_seg_load, ind_global_best.total_cost)
            
            # Improving phase
            #print("RTTP Improving phase")
            improvement = True
            while improvement:
                improvement = False
                random.shuffle(operators)
                for operator in operators:
                    task_ids = self.get_task_ids_from_sequence(ind_curr.sequence)
                    random.shuffle(task_ids)
                    for task_id in task_ids:
                        if operator != self.inverse:
                            task_id_candidate_list = self.construct_candidate_list(task_id, task_ids)
                            for task_id_2 in task_id_candidate_list:
                                task_id = task_id if task_id in ind_curr.sequence else self.tasks[task_id].inverse
                                task_id_2 = task_id_2 if task_id_2 in ind_curr.sequence else self.tasks[task_id_2].inverse
                                new_ind = operator(ind_curr, task_id, task_id_2)
                                if new_ind.total_cost < ind_global_best.total_cost:
                                    ind_global_best = new_ind
                                    ind_curr = new_ind
                                    improvement = True
                                    # temp
                                    if new_ind.total_cost < self.best_solution.total_cost and timer()-self.start_time <= self.time_limit:
                                        print(str(new_ind.total_cost)+"\t"+str(timer()-self.start_time)+"\t0")
                                    break
                        else:
                            new_ind = operator(ind_curr, task_id)
                            if new_ind.total_cost < ind_global_best.total_cost:
                                ind_global_best = new_ind
                                ind_curr = new_ind
                                improvement = True
                                # temp
                                if new_ind.total_cost < self.best_solution.total_cost and timer()-self.start_time <= self.time_limit:
                                    print(str(new_ind.total_cost)+"\t"+str(timer()-self.start_time)+"\t0")
            
            #print(ind_global_best.sequence, ind_global_best.route_seg_load, ind_global_best.total_cost)
            
            if ind_curr.total_cost < ind_local_best_total_cost:
                ind_local_best_total_cost = ind_curr.total_cost
                w = 0
            else:
                w += 1
        
        return ind_global_best
    
    
    def manage_penalty_parameter(self, penalty_parameter, feas_count, infeas_count, excess_dem):
        # infeasible solution
        if excess_dem != 0:
            feas_count += 1
            if infeas_count > 0:
                infeas_count = 0
        # feasible solution
        else:
            infeas_count += 1
            if feas_count > 0:
                feas_count = 0
        # halve the penalty parameter
        if feas_count == 5:
            penalty_parameter = penalty_parameter / 2
            feas_count = 0
        # double the penalty parameter
        elif infeas_count == 5:
            penalty_parameter = penalty_parameter * 2
            infeas_count = 0
        return penalty_parameter, feas_count, infeas_count
        
    
    # infeasible descent procedure
    def idp(self, ind):
        penalty_parameter = ind.total_cost / (2 * self.capacity)
        feas_count = 0
        infeas_count = 0
        
        best_ind = ind
        best_ind_cost = ind.total_cost
        
        # first stage
        #print("IDP first stage")
        task_ids = self.get_task_ids_from_sequence(ind.sequence)
        operators = [self.insert, self.double_insert, self.swap]
        for operator in operators:
            for task_id_1 in task_ids:
                for task_id_2 in task_ids:
                    if task_id_1 != task_id_2:
                        
                        new_ind = operator(ind, task_id_1, task_id_2, False)
                        excess_dem = self.calculate_excess_demand(new_ind.route_seg_load)
                        new_ind_cost = new_ind.total_cost + penalty_parameter * excess_dem                          
                        
                        if new_ind_cost < best_ind_cost:
                            best_ind = new_ind
                            best_ind_cost = new_ind_cost
                            
                            # temp
                            if new_ind.total_cost < self.best_solution.total_cost and \
                                excess_dem == 0 and \
                                timer()-self.start_time <= self.time_limit:
                                print(str(new_ind.total_cost)+"\t"+str(timer()-self.start_time)+"\t0")
                                                           
                            penalty_parameter, feas_count, infeas_count = \
                                self.manage_penalty_parameter(penalty_parameter, feas_count, infeas_count, excess_dem)
        
        #print(best_ind.sequence, best_ind.route_seg_load, best_ind.total_cost, best_ind_cost)
        
        # second stage
        improvement_found = False
        if best_ind.total_cost == ind.total_cost:
            #print("IDP second stage")
            r_nr = len(ind.route_seg)
            r_ids = [i for i in range(r_nr)]
            # possible combinations of route ids for merge-split operator
            possible_combinations = [[i] for i in range(r_nr)] + [r_ids]
            if r_nr > 2:
                for i in range(2,r_nr):
                    possible_combinations += combinations(r_ids, i)
            random.shuffle(possible_combinations)
            
            tried_combinations_nr = len(possible_combinations) if len(possible_combinations) < 100 else 100
            # try the first max. 100 possible combinations
            for i in range(tried_combinations_nr):
                # use merge-split
                new_ind = self.merge_split(best_ind, selected_route_ids=possible_combinations[i])
                
                excess_dem = self.calculate_excess_demand(new_ind.route_seg_load)
                new_ind_cost = new_ind.total_cost + penalty_parameter * excess_dem                          
                
                if new_ind_cost < best_ind_cost:
                    improvement_found = True
                    best_ind = new_ind
                    best_ind_cost = new_ind_cost
                    
                    # temp
                    if new_ind.total_cost < self.best_solution.total_cost and \
                        excess_dem == 0 and \
                        timer()-self.start_time <= self.time_limit:
                        print(str(new_ind.total_cost)+"\t"+str(timer()-self.start_time)+"\t0")                              
                    
                    penalty_parameter, feas_count, infeas_count = \
                        self.manage_penalty_parameter(penalty_parameter, feas_count, infeas_count, excess_dem)
            
            #print(best_ind.sequence, best_ind.route_seg_load, best_ind.total_cost, best_ind_cost)
                            
        if improvement_found:
            # first stage
            #print("IDP first stage")
            task_ids = self.get_task_ids_from_sequence(ind.sequence)
            operators = [self.insert, self.double_insert, self.swap]
            for operator in operators:
                for task_id_1 in task_ids:
                    for task_id_2 in task_ids:
                        if task_id_1 != task_id_2:
                            
                            new_ind = operator(ind, task_id_1, task_id_2, False)
                            excess_dem = self.calculate_excess_demand(new_ind.route_seg_load)
                            new_ind_cost = new_ind.total_cost + penalty_parameter * excess_dem                          
                            
                            if new_ind_cost < best_ind_cost:
                                best_ind = new_ind
                                best_ind_cost = new_ind_cost
                                
                                # temp
                                if new_ind.total_cost < self.best_solution.total_cost and \
                                    excess_dem == 0 and \
                                    timer()-self.start_time <= self.time_limit:
                                    print(str(new_ind.total_cost)+"\t"+str(timer()-self.start_time)+"\t0")
                                
                                penalty_parameter, feas_count, infeas_count = \
                                    self.manage_penalty_parameter(penalty_parameter, feas_count, infeas_count, excess_dem)
            #print(best_ind.sequence, best_ind.route_seg_load, best_ind.total_cost, best_ind_cost)
        
        return best_ind
                
    
    # calculate the distance for each task, sort it, keep only the top x
    
    def calculate_distance_between_two_tasks(self, task_id_1, task_id_2):
        sum_dist = self.min_cost[self.tasks[task_id_1].tail_node][self.tasks[task_id_2].head_node]
        dist_count = 1
        if self.tasks[task_id_1].inverse != None:
            sum_dist += self.min_cost[self.tasks[task_id_1].head_node][self.tasks[task_id_2].head_node]
            dist_count += 1
        if self.tasks[task_id_2].inverse != None:
            sum_dist += self.min_cost[self.tasks[task_id_1].tail_node][self.tasks[task_id_2].tail_node]
            dist_count += 1
        if  self.tasks[task_id_1].inverse != None and self.tasks[task_id_2].inverse != None:
            sum_dist += self.min_cost[self.tasks[task_id_1].head_node][self.tasks[task_id_2].tail_node]
            dist_count += 1
        
        return sum_dist / dist_count
    
    
    def construct_candidate_list(self, task_id, task_ids):
        csize = 12  # candidate list size
        task_ids.remove(task_id)
        dist_values = [inf for i in range(len(task_ids))]
        for index, other_task_id in enumerate(task_ids):
            dist_values[index] = (other_task_id, self.calculate_distance_between_two_tasks(task_id, other_task_id))
        
        return [t_id for t_id, dist_value in sorted(dist_values, key=lambda x: x[1])[:csize]]
    
    
    def hamming_distance(self, ind1, ind2):
        n = self.req_edge_num
        m = min(len(ind1.route_seg),len(ind2.route_seg))
        
        same_dh_link_nr = 0
        ind1_dh_links = set()
        
        for i in range(len(ind1.sequence)-1):
           ind1_dh_links.add((self.tasks[ind1.sequence[i]].head_node, \
                     self.tasks[ind1.sequence[i+1]].tail_node))
        
        for i in range(len(ind2.sequence)-1):
            if (self.tasks[ind2.sequence[i]].head_node, \
                     self.tasks[ind2.sequence[i+1]].tail_node) in ind1_dh_links:
                same_dh_link_nr += 1
        
        return n + m - same_dh_link_nr
    
    
    def calculate_quality_and_distance_fitness(self, ind_new):
        self.population.append(ind_new)
        new_population_size = self.population_size + 1
        
        AD_pop = np.zeros(new_population_size, dtype=int)
        for i in range(new_population_size):
            AD_pop[i] = 0
            for j in range(new_population_size):
                if i != j:
                    AD_pop[i] += self.hamming_distance(self.population[i], \
                                                      self.population[j])
            AD_pop[i] = AD_pop[i] / new_population_size - 1
        
        f_pop = [ind.total_cost for ind in self.population]
        
        OR = np.argsort(f_pop)
        DR = AD_pop.argsort()[::-1]
        
        population_qdf = np.zeros(new_population_size, dtype=int)       
        for i in range(new_population_size):
            population_qdf[OR[i]] += (i+1) * alpha
            population_qdf[DR[i]] += (i+1) * (1-alpha)
        
        max_qdf_value = max(population_qdf)
        max_qdf_value_index = list(population_qdf).index(max_qdf_value)
        del(self.population[max_qdf_value_index])
        
        if max_qdf_value_index == self.population_size:
            return False
        else:
            return True
    

# ------------------------------------ ACOPR --------------------------------------------


    # ACOPR main algorithm
    def acopr(self):
        # initialization
        start_time = timer()
        tau_matrice = np.full((len(self.task_ids_with_inv)+1, len(self.task_ids_with_inv)+1), tau_0, dtype=float)
        g_best_solution = Individual()
        i = 0
        for i in range(Iter):
            #print("Iteration "+str(i))
            prev_g_best_solution = copy.deepcopy(g_best_solution)
            i_best_solution = None
            for k in range(b):
                # CARP process
                solution = self.construct_solution(tau_matrice)
                if (i_best_solution == None) or (solution.total_cost < i_best_solution.total_cost):
                    i_best_solution = solution
                # local pheromone updating
                for id in range(len(solution.sequence)-1):
                    tau_matrice[solution.sequence[id]][solution.sequence[id+1]] = \
                    (rho * tau_matrice[solution.sequence[id]][solution.sequence[id+1]]) + ((1 - rho) * tau_0)
            # apply local search (2-opt, swap, insert)
            new_sols = [i_best_solution]
            new_sols.append(self.merge_split(i_best_solution))
            new_sols.append(self.two_opt(i_best_solution))
            new_sols.append(self.swap(i_best_solution))
            new_sols.append(self.insert(i_best_solution))
            new_sols.append(self.double_insert(i_best_solution))
            new_sols.append(self.inverse(i_best_solution))
            i_best_solution = min(new_sols, key=attrgetter('total_cost'))
            
            # apply path relinking (if there is a new global best solution)
            if g_best_solution.total_cost == inf:
                g_best_solution = copy.deepcopy(i_best_solution)
            if i_best_solution.total_cost < g_best_solution.total_cost:
                initial_solution = copy.deepcopy(g_best_solution)
                guiding_solution = copy.deepcopy(i_best_solution)
                
                # move the 0s in the initial solution, get ids of the "misplaced" tasks
                is_tid_seq = []
                initial_solution_mod = []
                mp_task_ids = set() # (task_id, index of the desired place)
                for tid in initial_solution.sequence:
                    if tid != 0:
                        is_tid_seq.append(tid)
                is_tid = 0
                for index in range(len(guiding_solution.sequence)):
                    if guiding_solution.sequence[index] != 0:
                        initial_solution_mod.append(is_tid_seq[is_tid])
                        if (guiding_solution.sequence[index] != is_tid_seq[is_tid]) or \
                            (self.tasks[guiding_solution.sequence[index]].inverse != is_tid_seq[is_tid]):
                            mp_task_ids.add((guiding_solution.sequence[index], index))
                        is_tid += 1
                    else:
                        initial_solution_mod.append(0)          
                
                new_g_best_solution = guiding_solution
                initial_solution_mod = self.ind_from_seq(initial_solution_mod)
                # print(initial_solution_mod.sequence)
                for task in mp_task_ids:
                    if task[0] in initial_solution_mod.sequence:
                        initial_solution_mod = \
                        self.swap(initial_solution_mod, task_id_1=task[0], task_id_2=initial_solution_mod.sequence[task[1]], only_feasible=False)
                    else:
                        initial_solution_mod = \
                        self.swap(initial_solution_mod, task_id_1=self.tasks[task[0]].inverse, task_id_2=initial_solution_mod.sequence[task[1]], only_feasible=False)
                    if (initial_solution_mod.total_cost < new_g_best_solution.total_cost) and \
                        (self.calculate_excess_demand(initial_solution_mod.route_seg_load) == 0):
                            new_g_best_solution = initial_solution_mod
                
                i_best_solution = new_g_best_solution
                g_best_solution = new_g_best_solution
            
            if g_best_solution.total_cost < prev_g_best_solution.total_cost:
                print(str(g_best_solution.total_cost)+"\t"+str(timer()-start_time)+"\t"+str(i))
            
            # global pheromone updating
            l_b = 1 / g_best_solution.total_cost
            l_s = 1 / i_best_solution.total_cost
            
            new_tau_matrice = rho * tau_matrice

            for id in range(len(i_best_solution.sequence)-1):
                new_tau_matrice[i_best_solution.sequence[id]][i_best_solution.sequence[id+1]] = \
                (rho * tau_matrice[i_best_solution.sequence[id]][i_best_solution.sequence[id+1]]) + ((1 - rho) * l_s)

            for id in range(len(g_best_solution.sequence)-1):
                new_tau_matrice[g_best_solution.sequence[id]][g_best_solution.sequence[id+1]] = \
                (rho * tau_matrice[g_best_solution.sequence[id]][g_best_solution.sequence[id+1]]) + ((1 - rho) * l_b)

            tau_matrice = copy.deepcopy(new_tau_matrice)
            
            #print(i_best_solution.sequence, i_best_solution.total_cost)
            #print(g_best_solution.sequence, g_best_solution.total_cost)
        return(g_best_solution)
    
    
    def construct_solution(self, tau_matrice):
        unserved_tasks = list(copy.deepcopy(self.task_ids_with_inv))
        seq = [0]
        route_load = 0
        while len(unserved_tasks) != 0:
            q = np.random.uniform()
            values = list(np.zeros((len(unserved_tasks),), dtype=float))
            next_task_id = 0
            for j in range(len(unserved_tasks)):
                dist = self.min_cost[self.tasks[seq[-1]].tail_node][self.tasks[unserved_tasks[j]].head_node]
                if dist <= 0:
                    dist = 1
                values[j] = tau_matrice[seq[-1]][unserved_tasks[j]] * ((1 / dist) ** beta)
            if q <= q_0:
                next_task_id = unserved_tasks[values.index(max(values))]
            if q > q_0:
                p = list(np.zeros((len(unserved_tasks),), dtype=float))
                values_sum = sum(values)
                if values_sum == 0:
                    values_sum = 0.001
                for j in range(len(unserved_tasks)):
                    p[j] = values[j] / values_sum
                next_task_id = random.choices(unserved_tasks, weights = p, k = 1)[0]
            if (route_load + self.tasks[next_task_id].demand) > self.capacity:
                seq.append(0)
                route_load = 0
            else:
                seq.append(next_task_id)
                unserved_tasks.remove(next_task_id)
                if self.tasks[next_task_id].inverse != None:
                    unserved_tasks.remove(self.tasks[next_task_id].inverse)
                route_load += self.tasks[next_task_id].demand
        seq.append(0)
        return self.ind_from_seq(seq)

# --------------------------------- Visualization --------------------------------------

    def draw_road_network(self):
        G = nx.DiGraph()
        for arc in self.arcs:
            task_id = self.find_task_id(arc.head_node, arc.tail_node)
            if task_id != -1:
                G.add_edge(arc.head_node, arc.tail_node, weight=arc.trav_cost, demand=self.tasks[task_id].demand)
            else:
                G.add_edge(arc.head_node, arc.tail_node, weight=arc.trav_cost, demand=0)
        pos = nx.circular_layout(G)
        plt.figure()
        nx.draw(
            G, pos, edge_color='black',
            #width=1, 
            linewidths=1,
            width = [1 if self.find_task_id(arc.head_node, arc.tail_node) == -1 else 2 for arc in self.arcs],
            node_size=500, node_color='black', alpha=1, font_color='white',
            labels={node: node for node in G.nodes()}
        )
        nx.draw_networkx_edge_labels(
            G, pos, verticalalignment='center_baseline',
            edge_labels= dict([((n1, n2), (d['weight'], d['demand'])) for n1, n2, d in G.edges(data=True)]),
            font_color='black'
        )
        plt.axis('off')
        plt.show()


    def draw_solution(self, solution_seq):
        colors = ["red","orange","green","blue","purple","olive"]
        route_plans = list()
        route_plans_arcs = list()
        route_plan = [0]
        for task_idx in range(1,len(solution_seq)):
            route_plan.append(solution_seq[task_idx])
            if solution_seq[task_idx] == 0:
                route_plans.append(route_plan)
                route_plan = [0]
        for route_plan in route_plans:
            route_plans_arcs.append(self.construct_routing_plan(route_plan))
        
        arc_colors = []
        G = nx.DiGraph()
        for arc in self.arcs:
            task_id = self.find_task_id(arc.head_node, arc.tail_node)
            if task_id != -1:
                G.add_edge(arc.head_node, arc.tail_node, weight=arc.trav_cost, demand=self.tasks[task_id].demand)
                for route_plan_index in range(len(route_plans)):
                    if task_id in route_plans[route_plan_index]:
                        arc_colors.append(colors[route_plan_index])
            else:
                G.add_edge(arc.head_node, arc.tail_node, weight=arc.trav_cost, demand=0)
                arc_colors.append('black')
        pos = nx.circular_layout(G)
        plt.figure()
        nx.draw(
            G, pos, #edge_color='black', width=1, 
            linewidths=1,
            edge_color = arc_colors,
            width = [1 if self.find_task_id(arc.head_node, arc.tail_node) == -1 else 2 for arc in self.arcs],
            node_size=500, node_color='black', alpha=1, font_color='white',
            labels={node: node for node in G.nodes()}
        )
        nx.draw_networkx_edge_labels(
            G, pos, verticalalignment='center_baseline',
            edge_labels= dict([((n1, n2), (d['weight'], d['demand'])) for n1, n2, d in G.edges(data=True)]),
            font_color='black'
        )
        plt.axis('off')
        plt.show()
