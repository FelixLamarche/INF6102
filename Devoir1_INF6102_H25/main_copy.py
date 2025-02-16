import os
import argparse
import solver_naive
import solver_advanced
import solver_advanced2
import test2_solver_advanced
import time
import time
from utils import Instance

def stop_program():
    print("Timeout reached, stopping the program.")
    raise SystemExit

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--agent', type=str, default='advanced')
    parser.add_argument('--infile', type=str, default='instances/trivial_1.txt')
    parser.add_argument('--viz-node-limit', type=int, default=4500)

    return parser.parse_args()

def process_instance(instance_file, agent, viz_node_limit):
    print(f"Processing instance: {instance_file}")

    instance = Instance(instance_file)

    print("***********************************************************")
    print("[INFO] Start the solving: Market study")
    print("[INFO] input file: %s" % instance.filepath)
    print("[INFO] number of nodes: %s" % (instance.N))
    print("[INFO] number of edges: %s" % (instance.M))
    print("***********************************************************")

    start_time = time.time()

    # Solving the problem using the selected agent
    if agent == "naive":
        solution = solver_naive.solve(instance)
    elif agent == "advanced":
        solution = test2_solver_advanced.solve(instance)
    elif agent == "advanced2":
        solution = solver_advanced2.solve(instance)
    else:
        raise Exception("This agent does not exist")

    solving_time = round((time.time() - start_time) / 60, 2)

    # Visualization if applicable
    # if viz_node_limit >= instance.N:
    #     instance.visualize_solution(solution)

    instance.save_solution(solution)
    cohesion, validity = instance.solution_value_and_validity(solution)
    unique_nodes = instance.unique_nodes_in_solution(solution)
    no_missing_added = instance.no_missing_or_added_nodes(solution)
    no_disconnected_groups = instance.no_disconnected_groups_in_solution(solution)

    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print(f"[INFO] Cohesion : {cohesion}")
    print(f"[INFO] Sanity check passed : {validity}\n\t Unique nodes in solution: {unique_nodes}\n\t No missing or added nodes : {no_missing_added}\n\t No disconnected groups : {no_disconnected_groups}")
    print("***********************************************************")


if __name__ == '__main__':
    args = parse_arguments()

    instances_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instances')
    print(instances_dir)

    # List all the instance files in the 'instances' directory
    instance_files = ['instances/'+f for f in os.listdir(instances_dir) if f.endswith('.txt')]
    print(instance_files)

    for instance_file in instance_files:
        #instance_file_path = os.path.join(args.instances_dir, instance_file)
        process_instance(instance_file, args.agent, args.viz_node_limit)