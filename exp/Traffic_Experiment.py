# Adapted from https://toruseo.jp/UXsim/docs/notebooks/demo_notebook_03en_pytorch.html

import numpy as np
import uxsim as ux
import itertools
import copy
import autogen
import autogen.trace as trace
from autogen.trace.optimizers import FunctionOptimizerV2

# Define minimum and maximum values for green light durations (parameter space)
MIN_GREEN_TIME = 15
MAX_GREEN_TIME = 90

# Define the simulation parameters (in seconds)
MAX_DURATION = 1800
SIMULATION_STEP = 6

# Define the demands
seed = 387
np.random.seed(seed)
demand = 0.25
demandDict = {}
for n1, n2 in itertools.permutations(['W1', 'E1', 'N1', 'S1'], 2):
    for t in range(0, MAX_DURATION, SIMULATION_STEP):
        demandDict[(n1,n2,t)] = np.random.uniform(0, demand)
# Add extra demand for E-W direction
for t in range(0, MAX_DURATION // 3, SIMULATION_STEP):
    demandDict[('W1','E1',t)] += demand
for t in range(2*MAX_DURATION // 3, MAX_DURATION, SIMULATION_STEP):
    demandDict[('E1','W1',t)] += demand

def create_world(EW_time, NS_time):
    """
    Creates a traffic intersection with the given green light durations for the East-West and North-South directions.
    """
    W = ux.World(
        name="Grid World",
        deltan=1,
        reaction_time=1,
        tmax=MAX_DURATION,
        print_mode=0, save_mode=0, show_mode=0,
        random_seed=seed,
        duo_update_time=120,
        show_progress=0,
        vehicle_logging_timestep_interval=-1
    )

    # network definition
    """
        N1
        |
    W1--I1--E1
        |
        S1
    """

    W1 = W.addNode("W1", -1, 0)
    E1 = W.addNode("E1", 1, 0)
    N1 = W.addNode("N1", 0, 1)
    S1 = W.addNode("S1", 0, -1)
    
    for k, v in demandDict.items():
        n1, n2, t = k
        node1 = eval(n1)
        node2 = eval(n2)
        W.adddemand(node1, node2, t, t+SIMULATION_STEP, v)

    I1 = W.addNode("I1", 0, 0, signal=[EW_time,NS_time])
    
    #E <-> W direction: signal group 0
    for n1,n2 in [[W1, I1], [I1, E1]]:
        W.addLink(n1.name+n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
        W.addLink(n2.name+n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
    #N <-> S direction: signal group 1
    for n1,n2 in [[N1, I1], [I1, S1]]:
        W.addLink(n1.name+n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)
        W.addLink(n2.name+n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)

    return W


def analyze_world(W):
    trips_completed = W.analyzer.od_trips_comp
    outputDict = {}
    for k,v in trips_completed.items():
        outputDict[k] = {'Volume': v}

    travel_free = W.analyzer.od_tt_free
    travel_actual = W.analyzer.od_tt_ave
    for k,v in travel_actual.items():
        delay = v - travel_free[k]
        outputDict[k]['Avg. Delay'] = v

    travel_variance = W.analyzer.od_tt_std
    for k,v in travel_variance.items():
        outputDict[k]['Trip Std.Dev.'] = v
    
    outputDict['OVERALL AVG. DELAY'] = W.analyzer.average_delay
    return outputDict


# 0. Sanity Check

for i in (MIN_GREEN_TIME, MAX_GREEN_TIME):
    W = create_world(i, i)
    W.exec_simulation()
    print(f"Green Time: {i}, Throughput: {analyze_world(W)}")


# 1. Heuristic: SCATS-like Control

# Aggregate E-W demand and N-S demand
EW_demand = 0
NS_demand = 0
for k, v in demandDict.items():
    n1, n2, t = k
    if (n1[0] in ['W', 'E']):
        EW_demand += v
    else:
        NS_demand += v

perstep_demand = (EW_demand + NS_demand)*SIMULATION_STEP / MAX_DURATION
cycle_length = int(60 // perstep_demand)
ratio = 1.0
if EW_demand > NS_demand:
    ratio = EW_demand / NS_demand
else:
    ratio = NS_demand / EW_demand
NS_time = MIN_GREEN_TIME
EW_time = MIN_GREEN_TIME
if EW_demand > NS_demand:
    NS_time = int(cycle_length // (1 + ratio))
    EW_time = cycle_length - NS_time
else:
    EW_time = int(cycle_length // (1 + ratio))
    NS_time = cycle_length - EW_time

if NS_time < MIN_GREEN_TIME:
    NS_time = MIN_GREEN_TIME
if EW_time < MIN_GREEN_TIME:
    EW_time = MIN_GREEN_TIME
if NS_time > MAX_GREEN_TIME:
    NS_time = MAX_GREEN_TIME
if EW_time > MAX_GREEN_TIME:
    EW_time = MAX_GREEN_TIME
W = create_world(EW_time, NS_time)
W.exec_simulation()
print(f"EW Green Time: {EW_time}, NS Green Time: {NS_time}, Result: {analyze_world(W)}")
benchmark = analyze_world(W)['OVERALL AVG. DELAY']

# 2. Blackbox Optimization: Gaussian Process

from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

EW_dim = Integer(MIN_GREEN_TIME, MAX_GREEN_TIME, name='EW_time')
NS_dim = Integer(MIN_GREEN_TIME, MAX_GREEN_TIME, name='NS_time')
dimensions = [EW_dim, NS_dim]

@use_named_args(dimensions=dimensions)
def f(EW_time, NS_time):
    if EW_time < MIN_GREEN_TIME:
        EW_time = MIN_GREEN_TIME
    if EW_time > MAX_GREEN_TIME:
        EW_time = MAX_GREEN_TIME

    if NS_time < MIN_GREEN_TIME:
        NS_time = MIN_GREEN_TIME
    if NS_time > MAX_GREEN_TIME:
        NS_time = MAX_GREEN_TIME

    W = create_world(EW_time, NS_time)
    W.exec_simulation()
    return analyze_world(W)['OVERALL AVG. DELAY']

res = gp_minimize(f,                  # the function to minimize
                  dimensions,      # the bounds on each dimension of x
                  n_calls=50,
                  n_random_starts=5,
                  verbose=True)
print(f"EW Green Time: {res.x[0]}, NS Green Time: {res.x[1]}, Result: {res.fun}")
plot_convergence(res)

# 3. Blackbox: Particle Swarm Optimization

import pyswarms as ps
import logging
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
from collections import deque

# Implementing a PSO for integer parameters; adapted from https://github.com/ljvmiranda921/pyswarms/issues/400

# Define custom operators
def compute_int_position(swarm, bounds, bh):
    """
    Custom position computation
    """
    try:
        temp_position = swarm.position.copy()
        temp_position += swarm.velocity

        if bounds is not None:
            temp_position = bh(temp_position, bounds)

        # This casting is the only change to the standard operator
        position = temp_position.astype(int)

    except AttributeError:
        print("Please pass a Swarm class")
        raise
    
    return position

def compute_int_velocity(swarm):
    try:
        swarm_size = swarm.position.shape
        c1 = swarm.options["c1"]
        c2 = swarm.options["c2"]
        w = swarm.options["w"]

        cognitive = (
                c1
                * np.random.uniform(0,1,swarm_size)
                * (swarm.pbest_pos - swarm.position)
        )
        social = (
                c2
                * np.random.uniform(0,1,swarm_size)
                * (swarm.best_pos - swarm.position)
        )

        # This casting is the only change to the standard operator
        updated_velocity = ((w * swarm.velocity) + cognitive + social).astype(int)

    except AttributeError:
        print("Please pass a Swarm class")
        raise

    return updated_velocity

# Define a custom topology. This is not 100% necessary, one could also use the
# built-in topologies. The following is the exact same as the Star topology
# but the compute_velocity and compute_position methods have been replaced
# by the custom ones
class IntStar(ps.backend.topology.Topology):
    def __init__(self, static=None, **kwargs):
        super(IntStar, self).__init__(static=static)

    def compute_gbest(self, swarm, **kwargs):
        try:
            if self.neighbor_idx is None:
                self.neighbor_idx = np.tile(
                        np.arange(swarm.n_particles), (swarm.n_particles, 1)
                        )
            if np.min(swarm.pbest_cost) < swarm.best_cost:
                best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
                best_cost = np.min(swarm.pbest_cost)
            else:
                best_pos, best_cost = swarm.best_pos, swarm.best_cost

        except AttributeError:
            print("Please pass a Swarm class")
            raise
        else:
            return best_pos, best_cost

    def compute_velocity(self, swarm):
        return compute_int_velocity(swarm)
    
    def compute_position(self, swarm, bounds, bh):
        return compute_int_position(swarm, bounds, bh)

# Define custom Optimizer class
class IntOptimizerPSO(ps.base.SwarmOptimizer):
    def __init__(self, n_particles, dimensions, options, bounds=None):
        super(IntOptimizerPSO, self).__init__(
                n_particles=n_particles,
                dimensions=dimensions,
                options=options,
                bounds=bounds,
                velocity_clamp=None,
                center=1.0,
                ftol=-np.inf,
                init_pos=None)
        self.reset()
        # The periodic strategy will leave the velocities on integer values
        self.bh = ps.backend.handlers.BoundaryHandler(strategy="periodic")
        self.top = IntStar()
        self.rep = ps.utils.Reporter(logger=logging.getLogger(__name__))
        self.name = __name__

    # More or less copy-paste of the optimize method of the GeneralOptimizerPSO
    def optimize(self, func, iters):
        self.bh.memory = self.swarm.position

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        for i in self.rep.pbar(iters, self.name):
            self.swarm.current_cost = ps.backend.operators.compute_objective_function(self.swarm, func, pool=None)
            self.swarm.pbest_pos, self.swarm.pbest_cost = ps.backend.operators.compute_pbest(self.swarm)
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                    self.swarm, **self.options
            )
            self.rep.hook(best_cost=self.swarm.best_cost)
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # You could also just use the custom operators on the next two lines
            self.swarm.velocity = self.top.compute_velocity(self.swarm) #compute_int_velocity(self.swarm)
            self.swarm.position = self.top.compute_position(self.swarm, self.bounds, self.bh) #compute_int_position(self.swarm, self.bounds, self.bh)
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
                self.swarm.pbest_cost.argmin()
        ].copy()
        self.rep.log(
                "Optimization finished | best cost: {}, best pos: {}".format(
                    final_best_cost, final_best_pos
        ))
        
        return final_best_cost, final_best_pos

# Define objective function with the cost decorator allows the defintion of the
# objective function for one particle
@ps.cost
def ps_f(X):
    EW_time = X[0]
    if EW_time < MIN_GREEN_TIME:
        EW_time = MIN_GREEN_TIME
    if EW_time > MAX_GREEN_TIME:
        EW_time = MAX_GREEN_TIME

    NS_time = X[1]
    if NS_time < MIN_GREEN_TIME:
        NS_time = MIN_GREEN_TIME
    if NS_time > MAX_GREEN_TIME:
        NS_time = MAX_GREEN_TIME

    W = create_world(EW_time, NS_time)
    W.exec_simulation()
    return analyze_world(W)['OVERALL AVG. DELAY']


opt = IntOptimizerPSO(n_particles = 5,
                      dimensions = 2,
                      options = {"c1": 0.5, "c2": 0.3, "w": 1.9},
                      bounds=(MIN_GREEN_TIME*np.ones(2), MAX_GREEN_TIME*np.ones(2))
                      )

c, p = opt.optimize(ps_f, 10)
print(f"EW Green Time: {p[0]}, NS Green Time: {p[1]}, Result: {c}")
plot_cost_history(cost_history=opt.cost_history)
plt.show()


#@trace.trace_op()
#def run_simulation(W, threshold):
#    """
#    Runs traffic simulation until the intersection has cars waiting greater than a threshold.
#    Returns the number of vehicles waiting at each link.
#    """
#    waiting_vehicles = None
#    while W.check_simulation_ongoing():
#        W.exec_simulation(duration_t = SIMULATION_STEP)
#        I1 = W.get_node('I1')
#        waiting_vehicles = 0
#        for l in I1.inlinks.values():
#            waiting_vehicles += l.num_vehicles_queue
#        if waiting_vehicles >= threshold:
#            break
#    print(W.T, waiting_vehicles)
#    return (W.T, waiting_vehicles)

@trace.trace_op(trainable=False, allow_external_dependencies=True)    
def traffic_simulation(EW_green_time, NS_green_time):
    """
    Runs a traffic simulation with the given green light durations for the East-West and North-South directions.
    A small green light duration for both directions will allow efficient flow.
    However, if the green light duration for a given direction is set too high, the vehicles in the other direction will experience a long delay.
    """
    W = create_world(EW_green_time, NS_green_time)
    #timestep, vehicles = run_simulation(W, 50)
    W.exec_simulation()

    return_dict = analyze_world(W)
    return return_dict

EW_x = trace.node(EW_time, trainable=True, constraint = f"[{MIN_GREEN_TIME},{MAX_GREEN_TIME}]")
NS_x = trace.node(NS_time, trainable=True, constraint = f"[{MIN_GREEN_TIME},{MAX_GREEN_TIME}]")
optimizer = FunctionOptimizerV2([EW_x, NS_x], config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))

optimizer.objective = 'You should suggest parameters so that the OVERALL AVG. DELAY is as small as possible.\n' + optimizer.default_objective

for i in range(50):
    result = traffic_simulation(EW_x, NS_x)
    target = result['OVERALL AVG. DELAY']
    feedback = 'OVERALL AVG. DELAY is too high.' if target > benchmark else 'OVERALL AVG. DELAY is competitive with baselines. Please try to optimize the intersection further.'
    
    optimizer.zero_feedback()
    optimizer.backward(target, feedback, visualize = True)
    optimizer.step(verbose = True)
