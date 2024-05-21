# Adapted from https://toruseo.jp/UXsim/docs/notebooks/demo_notebook_03en_pytorch.html

import numpy as np
import uxsim as ux
import itertools
import argparse
import os
import pickle
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import pyswarms as ps
import logging
import autogen
import autogen.trace as trace
from autogen.trace.optimizers import FunctionOptimizerV2Memory, OPRO
from autogen.trace.bundle import ExceptionNode


#####
# HELPER FUNCTIONS FOR IMPLEMENTING PARTICLE SWARM OPTIMIZATION WITH INTEGER PARAMETERS
# Adapted from https://github.com/ljvmiranda921/pyswarms/issues/400


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

        cognitive = c1 * np.random.uniform(0, 1, swarm_size) * (swarm.pbest_pos - swarm.position)
        social = c2 * np.random.uniform(0, 1, swarm_size) * (swarm.best_pos - swarm.position)

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
                self.neighbor_idx = np.tile(np.arange(swarm.n_particles), (swarm.n_particles, 1))
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
    def __init__(self, n_particles, dimensions, options, bounds=None, init_pos=None):
        super(IntOptimizerPSO, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=None,
            center=1.0,
            ftol=-np.inf,
            init_pos=None,
        )
        self.reset()
        # The periodic strategy will leave the velocities on integer values
        self.bh = ps.backend.handlers.BoundaryHandler(strategy="periodic")
        self.top = IntStar()
        self.rep = ps.utils.Reporter(logger=logging.getLogger(__name__))
        self.name = __name__

        #Ensure that if init_pos is passed, one of the swarm positions is the init_pos
        if init_pos is not None:
            self.swarm.position[0] = init_pos

    # More or less copy-paste of the optimize method of the GeneralOptimizerPSO
    def optimize(self, func, iters):
        self.bh.memory = self.swarm.position

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        for i in self.rep.pbar(iters, self.name):
            self.swarm.current_cost = ps.backend.operators.compute_objective_function(self.swarm, func, pool=None)
            self.swarm.pbest_pos, self.swarm.pbest_cost = ps.backend.operators.compute_pbest(self.swarm)
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm, **self.options)
            self.rep.hook(best_cost=self.swarm.best_cost)
            hist = self.ToHistory(
                best_cost=self.swarm.current_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # You could also just use the custom operators on the next two lines
            self.swarm.velocity = self.top.compute_velocity(self.swarm)  # compute_int_velocity(self.swarm)
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )  # compute_int_position(self.swarm, self.bounds, self.bh)
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        self.rep.log("Optimization finished | best cost: {}, best pos: {}".format(final_best_cost, final_best_pos))

        return final_best_cost, final_best_pos


# END OF PSO HELPER FUNCTIONS
#####


# Define minimum and maximum values for green light durations (parameter space)
MIN_GREEN_TIME = 15
MAX_GREEN_TIME = 90


# Define the simulation parameters (in seconds)
MAX_DURATION = 1800
SIMULATION_STEP = 6


# Define the demands
def create_demand(seed, demand=0.25):
    np.random.seed(seed)
    demandDict = {}
    for n1, n2 in itertools.permutations(["W1", "E1", "N1", "S1"], 2):
        for t in range(0, MAX_DURATION, SIMULATION_STEP):
            demandDict[(n1, n2, t)] = np.random.uniform(0, demand)
    # Add extra demand for E-W direction
    for t in range(0, MAX_DURATION // 3, SIMULATION_STEP):
        demandDict[("W1", "E1", t)] += demand
    for t in range(2 * MAX_DURATION // 3, MAX_DURATION, SIMULATION_STEP):
        demandDict[("E1", "W1", t)] += demand
    return demandDict


@trace.bundle(trainable=False, allow_external_dependencies=True)
def create_world(EW_time, NS_time):
    """
    Creates a traffic intersection with the given green light durations (variables: EW_time and NS_time)
    for the East-West and North-South directions respectively.

    The intersection layout is: W1 -- I1 -- E1 ; N1 -- I1 -- S1.
    That is, the intersection I1 has incoming links from W1, E1, N1, S1 as well as outgoing links to W1, E1, N1, S1.

    Traffic can flow from any incoming link to any outgoing link.
    After creating the intersection, the simulation is run for a fixed duration.
    The simulation then reports statistics recorded at the intersection about the traffic demand and delays.
    """
    global seed
    global demand_dict

    assert EW_time >= MIN_GREEN_TIME and EW_time <= MAX_GREEN_TIME, "EW_time out of bounds."
    assert NS_time >= MIN_GREEN_TIME and NS_time <= MAX_GREEN_TIME, "NS_time out of bounds."

    W = ux.World(
        name="Grid World",
        deltan=1,
        reaction_time=1,
        tmax=MAX_DURATION,
        print_mode=0,
        save_mode=0,
        show_mode=0,
        random_seed=seed,
        duo_update_time=120,
        show_progress=0,
        vehicle_logging_timestep_interval=-1,
    )

    W1 = W.addNode("W1", -1, 0)
    E1 = W.addNode("E1", 1, 0)
    N1 = W.addNode("N1", 0, 1)
    S1 = W.addNode("S1", 0, -1)

    for k, v in demand_dict.items():
        n1, n2, t = k
        node1 = eval(n1)
        node2 = eval(n2)
        W.adddemand(node1, node2, t, t + SIMULATION_STEP, v)

    I1 = W.addNode("I1", 0, 0, signal=[EW_time, NS_time])

    # E <-> W direction: signal group 0
    for n1, n2 in [[W1, I1], [I1, E1]]:
        W.addLink(n1.name + n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
        W.addLink(n2.name + n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
    # N <-> S direction: signal group 1
    for n1, n2 in [[N1, I1], [I1, S1]]:
        W.addLink(n1.name + n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)
        W.addLink(n2.name + n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)

    return W


@trace.bundle(trainable=False, allow_external_dependencies=True)
def analyze_world(W, verbose=True):
    """
    Analyzes the statistics recorded at traffic intersection. Returns a dictionary containing the following statistics:
    - Average delay computed across all vehicles that arrived at the intersection and subsequently completed their trip.
    - For each origin-destination pair (e.g. N1,S1) := the trips attempted, completed, and estimated time lost per vehicle.
    - Best-case estimate of the delay per vehicle due to the intersection. For any vehicles that did not complete their trip, their trip time is imputed as 1 + the maximum recorded time of that trip.
    - An aggregate score for the quality of the intersection. Lower scores are better. Two factors contribute to the score: the best-case estimated delay, and the variability in this estimate across each origin-destination pair.
    """

    assert not W.check_simulation_ongoing(), "Simulation has not completed."

    outputDict = {"Avg. Delay": W.analyzer.average_delay}
    time_lost = 0
    num_vehicles = 0

    for k, v in W.analyzer.od_trips.items():
        outputDict[k] = {"Trips attempted": v}
        num_vehicles += v
        outputDict[k]["Trips completed"] = W.analyzer.od_trips_comp[k]
        theoretical_minimum = W.analyzer.od_tt_free[k]
        observed_delay = np.sum(W.analyzer.od_tt[k] - theoretical_minimum)
        imputed_delay = (np.max(W.analyzer.od_tt[k]) + 1 - theoretical_minimum) * (v - len(W.analyzer.od_tt))
        time_lost += observed_delay + imputed_delay
        outputDict[k]["Time lost per vehicle"] = (observed_delay + imputed_delay) / v

    outputDict["Best-Case Estimated Delay"] = time_lost / num_vehicles
    variance = 0
    for k, v in W.analyzer.od_trips.items():
        variance += ((outputDict[k]["Time lost per vehicle"] - outputDict["Best-Case Estimated Delay"]) ** 2) * v

    score = outputDict["Best-Case Estimated Delay"] + np.sqrt(variance / num_vehicles)
    outputDict["OVERALL SCORE"] = score

    if not verbose:
        for k in list(outputDict.keys()):
            if k != "OVERALL SCORE":
                del outputDict[k]
    return outputDict


def run_approach(method, num_iter, trace_memory=0, trace_config="OAI_CONFIG_LIST"):
    W = None
    return_val = None
    if method == "SanityCheck":
        return_val = np.zeros((2, 3))

        for i in (MIN_GREEN_TIME, MAX_GREEN_TIME):
            W = create_world(i, i)
            W.data.exec_simulation()
            result = analyze_world(W.data).data["OVERALL SCORE"]
            return_val[i // MAX_GREEN_TIME] = (i, i, result)
        return return_val

    elif method == "SCATS":
        return_val = np.zeros((1, 3))

        # Aggregate E-W demand and N-S demand
        EW_demand = 0
        NS_demand = 0
        for k, v in demand_dict.items():
            n1 = k[0]
            if n1[0] in ["W", "E"]:
                EW_demand += v
            else:
                NS_demand += v

        # Heuristic to decide cycle length: expected queue should not exceed 60
        perstep_demand = (EW_demand + NS_demand) * SIMULATION_STEP / MAX_DURATION
        cycle_length = int(60 // perstep_demand)

        # Heuristic to decide each green light duration: proportional to demand
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

        # Ensure green light durations are within bounds
        if NS_time < MIN_GREEN_TIME:
            NS_time = MIN_GREEN_TIME
        if EW_time < MIN_GREEN_TIME:
            EW_time = MIN_GREEN_TIME
        if NS_time > MAX_GREEN_TIME:
            NS_time = MAX_GREEN_TIME
        if EW_time > MAX_GREEN_TIME:
            EW_time = MAX_GREEN_TIME

        W = create_world(EW_time, NS_time)
        W.data.exec_simulation()
        result = analyze_world(W.data).data["OVERALL SCORE"]
        return_val[0] = (EW_time, NS_time, result)
        return return_val

    elif method == "GP":
        return_val = np.zeros((num_iter, 3))

        EW_dim = Integer(MIN_GREEN_TIME, MAX_GREEN_TIME, name="EW_time")
        NS_dim = Integer(MIN_GREEN_TIME, MAX_GREEN_TIME, name="NS_time")
        dimensions = [EW_dim, NS_dim]

        @use_named_args(dimensions=dimensions)
        def gp_f(EW_time, NS_time):
            W = create_world(EW_time, NS_time)
            W.data.exec_simulation()
            return analyze_world(W.data).data["OVERALL SCORE"]

        res = gp_minimize(
            gp_f,  # the function to minimize
            dimensions,  # the bounds on each dimension of x
            n_calls=num_iter,
            n_initial_points=trace_memory if num_iter > trace_memory else num_iter - 1,
            initial_point_generator="sobol",
            x0 = [MIN_GREEN_TIME, MIN_GREEN_TIME],  # initial point
            verbose=True,
        )
        for i in range(num_iter):
            return_val[i] = (res.x_iters[i][0], res.x_iters[i][1], res.func_vals[i])
        return return_val

    elif method == "PSO":
        return_val = np.zeros((num_iter*trace_memory, 3))

        @ps.cost
        def ps_f(X):
            EW_time = X[0]
            NS_time = X[1]
            W = create_world(EW_time, NS_time)
            W.data.exec_simulation()
            return analyze_world(W.data).data["OVERALL SCORE"]

        opt = IntOptimizerPSO(
            n_particles=trace_memory,
            dimensions=2,
            options={"c1": 0.5, "c2": 0.3, "w": 1.9},
            bounds=(MIN_GREEN_TIME * np.ones(2), MAX_GREEN_TIME * np.ones(2)),
            init_pos=[MIN_GREEN_TIME, MIN_GREEN_TIME],
        )

        c, p = opt.optimize(ps_f, num_iter)
        for i in range(num_iter):
            for j in range(trace_memory):
                return_val[i*trace_memory+j] = (opt.pos_history[i][j][0], opt.pos_history[i][j][1], opt.cost_history[i][j])
        return return_val

    elif method.startswith("Trace") or method.startswith("OPRO"):
        return_val = np.zeros((num_iter, 3))
        verbosity = "Verbose" in method

        def traffic_simulation(EW_green_time, NS_green_time):
            W = None
            try:
                W = create_world(EW_green_time, NS_green_time)
            except Exception as e:
                e_node = ExceptionNode(
                    e,
                    inputs={"EW_green_time": EW_green_time, "NS_green_time": NS_green_time},
                    description="[exception] Simulation raises an exception with these inputs.",
                    name="exception_step",
                )
                return e_node
            W.data.exec_simulation()
            return_dict = analyze_world(W, verbosity)
            return return_dict

        EW_x = trace.node(MIN_GREEN_TIME, trainable=True, constraint=f"[{MIN_GREEN_TIME},{MAX_GREEN_TIME}]")
        NS_x = trace.node(MIN_GREEN_TIME, trainable=True, constraint=f"[{MIN_GREEN_TIME},{MAX_GREEN_TIME}]")
        if method.startswith("OPRO"):
            optimizer = OPRO([EW_x, NS_x], memory_size=trace_memory, config_list=autogen.config_list_from_json(trace_config))    
        else:
            optimizer = FunctionOptimizerV2Memory(
                [EW_x, NS_x], memory_size=trace_memory, config_list=autogen.config_list_from_json(trace_config)
            )

        optimizer.objective = (
                "You should suggest values for the variables so that the OVERALL SCORE is as small as possible.\n"
                + "There is a trade-off in setting the green light durations.\n"
                + "If the green light duration for a given direction is set too low, then vehicles will queue up over time and experience delays, thereby lowering the score for the intersection.\n"
                + "If the green light duration for a given direction is set too high, vehicles in the other direction will queue up and experience delays, thereby lowering the score for the intersection.\n"
                + "The goal is to find a balance for each direction (East-West and North-South) that minimizes the overall score of the intersection.\n"
                + optimizer.default_objective
        )

        for i in range(num_iter):
            result = traffic_simulation(EW_x, NS_x)
            feedback = None
            if isinstance(result, ExceptionNode):
                return_val[i] = (EW_x.data, NS_x.data, np.inf)
                feedback = result.data
            else:
                return_val[i] = (EW_x.data, NS_x.data, result.data["OVERALL SCORE"])
                feedback = (
                    "OVERALL SCORE: "
                    + str(result.data["OVERALL SCORE"])
                    + "\nPlease try to optimize the intersection further. If you are certain that you have found the optimal solution, please suggest it again."
                )

            optimizer.zero_feedback()
            optimizer.backward(result, feedback, visualize=True)
            if "Mask" in method:
                optimizer.step(verbose=True, mask = ["#Documentation", "#Code", "#Inputs", "#Others"])
            else:
                optimizer.step(verbose=True)
        return return_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic simulation experiment.")
    parser.add_argument("--replications", type=int, default=20, help="Number of replications.")
    parser.add_argument("--demand", type=float, default=0.25, help="Demand for traffic simulation.")
    parser.add_argument("--trace_mem", type=int, default=5, help="Memory for trace optimization.")
    parser.add_argument(
        "--trace_config", type=str, default="OAI_CONFIG_LIST", help="Configuration file for trace optimization."
    )
    parser.add_argument("--iter", type=int, default=50, help="Number of iterations for optimization methods.")
    parser.add_argument("--output_prefix", type=str, default="results_", help="Output file for results.")

    args = parser.parse_args()

    global seed
    global demand_dict

    results = []
    for i in range(args.replications):
        seed = 42 + i
        demand_dict = create_demand(seed, args.demand)

        pkled_dict = None
        if os.path.exists(args.output_prefix + str(i)):
            pkl = open(args.output_prefix + str(i), "rb")
            pkled_dict = pickle.load(pkl)
            pkl.close()
        else:
            returned_val = run_approach('SCATS', args.iter, args.trace_mem, args.trace_config)
            pkled_dict = {'SCATS': returned_val}

            returned_val = run_approach('GP', args.iter, args.trace_mem, args.trace_config)
            pkled_dict['GP'] = returned_val

            returned_val = run_approach('PSO', args.iter // args.trace_mem, args.trace_mem, args.trace_config)
            pkled_dict['PSO'] = returned_val

            returned_val = run_approach('TraceVerbose', args.iter, args.trace_mem, args.trace_config)
            pkled_dict['Trace'] = returned_val

            returned_val = run_approach('OPROVerbose', args.iter, args.trace_mem, args.trace_config)
            pkled_dict['OPRO'] = returned_val

            returned_val = run_approach('TraceMaskVerbose', args.iter, args.trace_mem, args.trace_config)
            pkled_dict['TraceMask'] = returned_val

            returned_val = run_approach('Trace', args.iter, args.trace_mem, args.trace_config)
            pkled_dict['TraceScalar'] = returned_val

            if args.trace_mem > 0:
                returned_val = run_approach('TraceVerbose', args.iter, 0, args.trace_config)
                pkled_dict['TraceNoMem'] = returned_val

                returned_val = run_approach('TraceMaskVerbose', args.iter, 0, args.trace_config)
                pkled_dict['TraceNoMemScalar'] = returned_val

            pkl = open(args.output_prefix + str(i), "wb")
            pickle.dump(pkled_dict, pkl)
            pkl.close()

        results.append(pkled_dict)

    import matplotlib.pyplot as plt

    x_axis_len = results[0]["GP"].shape[0]
    x_axis = range(x_axis_len)
    #plt.axis([0, x_axis_len, 30, 200])

    def extract_mean_ste(result, method):
        method_results = np.zeros((args.replications, x_axis_len))
        for i in range(args.replications):
            method_results[i, :] = result[i][method][:, 2]
        # If nan or inf values are present, replace them with nan
        method_results[np.isnan(method_results)] = np.nan
        method_results[np.isinf(method_results)] = np.nan

        mean_results = np.nanmean(method_results, axis=0)
        std_results = np.nanstd(method_results, axis=0)
        non_nan = np.count_nonzero(~np.isnan(method_results), axis=0)
        ste_results = std_results / np.sqrt(non_nan)
        return mean_results, ste_results

    mean_scats, ste_scats = extract_mean_ste(results, "SCATS")
    plt.plot(x_axis, mean_scats, label="SCATS")
    plt.fill_between(x_axis, mean_scats - ste_scats, mean_scats + ste_scats, alpha=0.2)

    #mean_gp, ste_gp = extract_mean_ste(results, "GP")
    #plt.plot(x_axis, mean_gp, label="GP")
    #plt.fill_between(x_axis, mean_gp - ste_gp, mean_gp + ste_gp, alpha=0.2)

    #mean_pso, ste_pso = extract_mean_ste(results, "PSO")
    #plt.plot(x_axis, mean_pso, label="PSO")
    #plt.fill_between(x_axis, mean_pso - ste_pso, mean_pso + ste_pso, alpha=0.2)

    mean_trace, ste_trace = extract_mean_ste(results, "Trace")
    plt.plot(x_axis, mean_trace, label="Trace")
    plt.fill_between(x_axis, mean_trace - ste_trace, mean_trace + ste_trace, alpha=0.2)

    #mean_opro, ste_opro = extract_mean_ste(results, "OPRO")
    #plt.plot(x_axis, mean_opro, label="OPRO")
    #plt.fill_between(x_axis, mean_opro - ste_opro, mean_opro + ste_opro, alpha=0.2)

    mean_tracem, ste_tracem = extract_mean_ste(results, "TraceMask")
    plt.plot(x_axis, mean_tracem, label="TraceMasked")
    plt.fill_between(x_axis, mean_tracem - ste_tracem, mean_tracem + ste_tracem, alpha=0.2)

    mean_traces, ste_traces = extract_mean_ste(results, "TraceScalar")
    plt.plot(x_axis, mean_traces, label="TraceScalar")
    plt.fill_between(x_axis, mean_traces - ste_traces, mean_traces + ste_traces, alpha=0.2)

    #if "TraceNoMem" in results[0]:
    #    mean_trace_nomem, ste_trace_nomem = extract_mean_ste(results, "TraceNoMem")
    #    plt.plot(x_axis, mean_trace_nomem, label="TraceNoMem")
    #    plt.fill_between(x_axis, mean_trace_nomem - ste_trace_nomem, mean_trace_nomem + ste_trace_nomem, alpha=0.2)

    #    mean_trace_nomems, ste_trace_nomems = extract_mean_ste(results, "TraceNoMemScalar")
    #    plt.plot(x_axis, mean_trace_nomems, label="TraceNoMemScalar")
    #    plt.fill_between(x_axis, mean_trace_nomems - ste_trace_nomems, mean_trace_nomems + ste_trace_nomems, alpha=0.2)

    plt.title("Traffic Optimization -- GPT4-0125-Preview")
    plt.legend()
    plt.show()
