import utils as u

def execute_rr1(new_instance, ind_w_nr_seq, ind_wo_nr_seq, tasks):
    """Execute RR1 algorithm and return the solution."""
    print("\nRunning RR1...")
    best_solution_rr1 = new_instance.reroute_one_route(ind_w_nr_seq, ind_wo_nr_seq, tasks)
    best_solution_rr1 = new_instance.ind_from_seq(best_solution_rr1.sequence)
    print(f"RR1 Solution: {best_solution_rr1.sequence}\n"
          f"Route Segment Load: {best_solution_rr1.route_seg_load}\n"
          f"Total Cost: {best_solution_rr1.total_cost}")
    return best_solution_rr1

def execute_abc(new_instance, initial_solution):
    """Execute ABC algorithm."""
    print("\nRunning ABC...")
    best_solution_abc = new_instance.abc(initial_solution=initial_solution)
    print(f"ABC Solution: {best_solution_abc.sequence}\n"
          f"Route Segment Load: {best_solution_abc.route_seg_load}\n"
          f"Total Cost: {best_solution_abc.total_cost}")

def execute_hma(new_instance):
    """Execute HMA algorithm."""
    print("\nRunning HMA...")
    best_solution_hma = new_instance.hma()
    print(f"HMA Solution: {best_solution_hma.sequence}\n"
          f"Route Segment Load: {best_solution_hma.route_seg_load}\n"
          f"Total Cost: {best_solution_hma.total_cost}")

def run_scenario(instance, event_func, params, description):
    """
    Generic scenario runner for different events.
    
    Parameters:
        instance: The CARP instance.
        event_func: Event function to call (`event_task_appearance`, `event_increased_demand`, `event_vehicle_breakdown`).
        params: Parameters for the event function.
        description: Description of the scenario being tested.
    """
    print(f"\n\n--- {description} ---")
    log = instance.generate_travel_service_log(ind, 300, 550, params[0])  # Common parameters
    result = event_func(ind, log, *params[1:])
    
    # Unpack results based on scenario type
    if result is not None:
        if len(result) == 3:  # For new task or increased demand
            new_instance, (ind_w_nr_seq, ind_wo_nr_seq, tasks) = result
        else:  # For vehicle breakdown
            new_instance, reroute, (ind_w_nr_seq, ind_wo_nr_seq, tasks) = result
            if reroute != 1:
                print("No reroute needed.")
                return
        
        # Execute algorithms
        rr1_solution = execute_rr1(new_instance, ind_w_nr_seq, ind_wo_nr_seq, tasks)
        new_instance.prepare_tasks()
        execute_abc(new_instance, rr1_solution)
        execute_hma(new_instance)
    else:
        print("No valid instance created for this event.")

# Test case: egl-e1-A (EGL)
initial_instance = u.Instance()
initial_instance.import_from_file("input/egl-e1-A.dat")
initial_instance.calculate_shortest_paths()

# Initial solution
seq = [0, 4, 5, 6, 7, 59, 9, 10, 11, 12, 32, 82, 80, 79, 75, 0,
       97, 88, 100, 64, 14, 15, 16, 50, 81, 27, 77, 76, 51, 0,
       99, 17, 18, 20, 19, 21, 22, 84, 0,
       3, 47, 96, 86, 85, 23, 0,
       1, 2, 87, 38, 40, 41, 42, 44, 94, 39, 0]
initial_instance.route_to_vehicle = [0, 1, 2, 3, 4]
initial_instance.virtual_task_ids = [0, 0, 0, 0, 0]
initial_instance.free_vehicles = set()
ind = initial_instance.ind_from_seq(seq)

# Define events (parameters)
# - task appearance (crc, nt_arc, nt_demand, nt_serv_cost)
# - demand increased (crc, dit_arc, dit_dem_inc, dit_sc_inc)
# - vehicle breakdown (crc, vb_id)

scenarios = {
    "task appearance": {
        "function": initial_instance.event_task_appearance,
        "params": [(315, (25, 75), 58, 16), (315, (46, 45), 67, 12), (356, (43, 42), 11, 11), \
            (379, (42, 57), 20, 14), (406, (2, 1), 62, 32), (409, (73, 74), 40, 25), \
            (419, (9, 8), 37, 26), (427, (32, 31), 5, 58), (436, (24, 22), 64, 4), \
            (439, (15, 14), 99, 7), (457, (6, 5), 46, 8), (490, (41, 40), 207, 9), \
            (517, (22, 75), 66, 24), (520, (21, 51), 89, 2), (522, (39, 35), 67, 7)]
    },
    "demand increased": {
        "function": initial_instance.event_increased_demand,
        "params": [(326, (32, 34), 36, 36), (344, (54, 52), 11, 11), (345, (50, 52), 15, 15), \
             (374, (52, 54), 9, 9), (376, (68, 66), 32, 32), (384, (44, 45), 18, 18), \
             (415, (46, 47), 9, 9), (431, (44, 59), 11, 11), (449, (32, 35), 12, 12), \
             (468, (35, 32), 65, 65), (490, (44, 46), 2, 2), (490, (32, 33), 28, 28), \
             (493, (59, 44), 5, 5), (516, (35, 32), 24, 24), (540, (35, 41), 13, 13)]
        ]
    },
    "vehicle breakdown": {
        "function": initial_instance.event_vehicle_breakdown,
        "params": [(305, 2), (311, 1), (342, 2), (344, 0), (364, 0), (399, 4), \
            (451, 2), (463, 0), (490, 2), (495, 2), (506, 0), (507, 1), \
            (523, 2), (540, 2), (430, 1)]
        ]
    }
}

# Run scenarios
for scenario_name, scenario_data in scenarios.items():
    func = scenario_data["function"]
    for i, params in enumerate(scenario_data["params"], start=1):
        description = f"{scenario_name} Example {i}"
        run_scenario(initial_instance, func, params, description)
