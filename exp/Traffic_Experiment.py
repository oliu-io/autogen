# Adapted from https://toruseo.jp/UXsim/docs/notebooks/demo_notebook_03en_pytorch.html

import numpy as np
import uxsim as ux
import itertools
import copy

# Define minimum and maximum values for green light durations
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 120
MAX_DURATION = 3600
SIMULATION_STEP = 30

# demand definition
seed = 387
np.random.seed(seed)
demand = 0.21
demandDict = {}
for n1, n2 in itertools.permutations(['W1', 'W2', 'E1', 'E2', 'N1', 'N2', 'S1', 'S2'], 2):
    for t in range(0, MAX_DURATION, SIMULATION_STEP):
        demandDict[(n1,n2,t)] = np.random.uniform(0, demand)

def create_world(EW_time, NS_time):
    W = ux.World(
        name="Grid World",
        deltan=5,
        tmax=MAX_DURATION,
        print_mode=1, save_mode=0, show_mode=0,
        random_seed=seed,
        duo_update_time=600
    )

    # network definition
    """
        N1  N2
        |   |
    W1--I1--I2--E1
        |   |
    W2--I3--I4--E2
        |   |
        S1  S2
    """

    W1 = W.addNode("W1", -1, 0)
    W2 = W.addNode("W2", -1, -1)
    E1 = W.addNode("E1", 2, 0)
    E2 = W.addNode("E2", 2, -1)
    N1 = W.addNode("N1", 0, 1)
    N2 = W.addNode("N2", 1, 1)
    S1 = W.addNode("S1", 0, -2)
    S2 = W.addNode("S2", 1, -2)

    for k, v in demandDict.items():
        n1, n2, t = k
        node1 = eval(n1)
        node2 = eval(n2)
        W.adddemand(node1, node2, t, t+SIMULATION_STEP, v)

    I1 = W.addNode("I1", 0, 0, signal=[EW_time,NS_time])
    I2 = W.addNode("I2", 1, 0, signal=[EW_time,NS_time])
    I3 = W.addNode("I3", 0, -1, signal=[EW_time,NS_time])
    I4 = W.addNode("I4", 1, -1, signal=[EW_time,NS_time])

    #E <-> W direction: signal group 0
    for n1,n2 in [[W1, I1], [I1, I2], [I2, E1], [W2, I3], [I3, I4], [I4, E2]]:
        W.addLink(n1.name+n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
        W.addLink(n2.name+n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
    #N <-> S direction: signal group 1
    for n1,n2 in [[N1, I1], [I1, I3], [I3, S1], [N2, I2], [I2, I4], [I4, S2]]:
        W.addLink(n1.name+n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)
        W.addLink(n2.name+n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)

    return W

# SANITY CHECK
#W1 = create_world(MAX_GREEN_TIME, MAX_GREEN_TIME)
#W1.exec_simulation()
#W1.analyzer.print_simple_stats()

#W2 = create_world(MIN_GREEN_TIME, MIN_GREEN_TIME)
#W2.exec_simulation()
#W2.analyzer.print_simple_stats()

import autogen
import autogen.trace as trace
from autogen.trace.optimizers import FunctionOptimizer

#@trace.trace_op(trainable=False, n_outputs=1)
def run_simulation(W, threshold):
    """
    Runs simulation until the number of waiting vehicles exceeds the threshold.
    Returns the number of vehicles waiting at each link.
    """
    while W.check_simulation_ongoing():
        W.exec_simulation(duration_t = SIMULATION_STEP)
        vehicles = {}
        I1 = W.get_node('I1')
        for l in I1.inlinks.values():
            vehicles[l] = l.num_vehicles_queue
        I2 = W.get_node('I2')
        for l in I2.inlinks.values():
            vehicles[l] = l.num_vehicles_queue
        I3 = W.get_node('I3')
        for l in I3.inlinks.values():
            vehicles[l] = l.num_vehicles_queue
        I4 = W.get_node('I4')
        for l in I4.inlinks.values():
            vehicles[l] = l.num_vehicles_queue
        waiting_vehicles = max(vehicles.values())
        if waiting_vehicles > threshold:
            return vehicles
    return None
    
@trace.trace_op(trainable=False, n_outputs=1)
def traffic_simulation(green_time):
    """
    Road network is a grid with 4 intersections.
        N1  N2
        |   |
    W1--I1--I2--E1
        |   |
    W2--I3--I4--E2
        |   |
        S1  S2
    Each intersection has 2 incoming and 2 outgoing links.
    Traffic lights are controlled by a fixed-time controller.
    Green light duration is set to green_time which is the input to this function.
    """
    W = create_world(green_time, green_time)

    init_interrupt_threshold = 50 # initial threshold for interrupting simulation
    vehicles = run_simulation(W, init_interrupt_threshold)
    returned_info = []
    while vehicles is not None:
        returned_info.append((W.TIME, vehicles))
        init_interrupt_threshold *= 2
        vehicles = run_simulation(W, init_interrupt_threshold)
    return (W, returned_info)

def scorer(world_data, vehicle_data):
    # Use vehicle_data to specify whether N/S is getting clogged more or E/W
    feedback = ""
    world_data.analyzer.basic_analysis()
    feedback += f"Average Delay: {world_data.analyzer.average_delay}\n"

    for item in vehicle_data:
        time, vehicles = item
        feedback += f"Waiting Vehicles at Time: {time}\n"
        for link, num_vehicles in vehicles.items():
            feedback += f"{link.name}: {num_vehicles}\n"
    return feedback

x = trace.node(int(0.5*(MIN_GREEN_TIME+MAX_GREEN_TIME)), trainable=True, constraint = f"[{MIN_GREEN_TIME},{MAX_GREEN_TIME}]")
optimizer = FunctionOptimizer([x], config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))

result = traffic_simulation(x)
feedback = scorer(result.data[0], result.data[1])
    
optimizer.zero_feedback()
optimizer.backward(result, feedback, visualize = True)
optimizer.step(verbose = True)
