import utils as u

# Configuration
input_files = [                     # list of the input files
    "input/kshs1.dat",
    "input/egl-e1-A.dat",
    "input/egl-s1-A.dat",
    "input/egl-g1-A.dat",
    "input/egl-g2-A.dat"
]
selected_file_index = 0             # CARP instance
use_abc = True                      # use CARP-ABC algorithm?
use_acopr = True                    # use ACOPR algorithm?
use_hma = True                      # use HMA algorithm?
num_runs = 30                       # number of runs

# Initialize instance and load data
instance = u.Instance()
input_file_path = input_files[selected_file_index]
instance.import_from_file(input_file_path)
instance.calculate_shortest_paths()

def run_algorithm(algorithm_name, algorithm_func):
    """Runs the specified algorithm and prints its results."""
    best_solution = algorithm_func()
    print(f"\n{algorithm_name} Solution:")
    print(f"  Sequence: {best_solution.sequence}")
    print(f"  Route Segment Load: {best_solution.route_seg_load}")
    print(f"  Total Cost: {best_solution.total_cost}")

# Run experiments
for i in range(num_runs):
    print(f"\n=== Run {i+1} ===")
    
    # Run selected algorithms
    if use_abc:
        run_algorithm("ABC", instance.abc)
    if use_acopr:
        run_algorithm("ACOPR", instance.acopr)
    if use_hma:
        run_algorithm("HMA", instance.hma)
