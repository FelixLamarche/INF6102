import os
import argparse
import solver_naive
import solver_advanced
import time
from utils import Instance

if __name__ == '__main__':

    # define paths
    base_folder = "instances"

    # Iterate over each test folder, objective
    for test_folder in os.listdir(base_folder):
        print(test_folder)
        # test_path = os.path.join(base_folder, test_folder)
        # test = os.path.abspath(test_path)
        # print(test)
        # print(os.path.isdir(test))
        # if os.path.isdir(test_path):

        instance = Instance(test_folder)
        print("***********************************************************")
        print("[INFO] Start the solving: Market study")
        print("[INFO] input file: %s" % instance.filepath)
        print("[INFO] number of nodes: %s" % (instance.N))
        print("[INFO] number of edges: %s" % (instance.M))
        print("***********************************************************")

        solution = solver_advanced.solve(instance)
        
        if args.viz_node_limit >= instance.N:
            instance.visualize_solution(solution)

        instance.save_solution(solution)
        cohesion, validity = instance.solution_value_and_validity(solution)
        unique_nodes = instance.unique_nodes_in_solution(solution)
        no_missing_added = instance.no_missing_or_added_nodes(solution)
        no_disconnected_groups = instance.no_disconnected_groups_in_solution(solution)

        print("***********************************************************")
        print("[INFO] Solution obtained")
        print(f"[INFO] Cohesion : {cohesion}")
        print(f"[INFO] Sanity check passed : {validity}\n\t Unique nodes in solution: {unique_nodes}\n\t No missing or added nodes : {no_missing_added}\n\t No disconnected groups : {no_disconnected_groups}")
        print("***********************************************************")
