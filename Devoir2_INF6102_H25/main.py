import argparse
import solver_naive
import solver_advanced
import time
from utils import Instance, Solution


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--agent', type=str, default='naive')
    parser.add_argument('--infile', type=str, default='./instances/trivial.txt')
    parser.add_argument('--no-viz', default=False, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    instance = Instance(args.infile)
    no_viz = args.no_viz

    print("***********************************************************")
    print("[INFO] Start the solving: Linking facilities")
    print("[INFO] input file: %s" % instance.filepath)
    print("[INFO] number of nodes: %s" % (instance.N))
    print("[INFO] number of edges: %s" % (instance.M))
    print("[INFO] number of profitable nodes: %s" % (instance.P))
    print("[INFO] Maximum budget: %s" % (instance.B))
    print("[INFO] Maximal distance: %s" % (instance.H))
    print("***********************************************************")

    start_time = time.time()

    if args.agent == "naive":
        # Reach all possible facilities
        solution: Solution = solver_naive.solve(instance)
    elif args.agent == "advanced":
        # Your nice agent
        solution = solver_advanced.solve(instance)
    else:
        raise Exception("This agent does not exist")


    solving_time = round((time.time() - start_time) / 60,2)

    if not no_viz:
        instance.visualize_solution(solution)

    instance.save_solution(solution)
    
    revenue, validity = instance.solution_value_and_validity(solution)
    consummed = instance.solution_cost(solution)
    no_added_edges = instance.no_added_edges(solution)
    all_linked_to_root = instance.all_edge_linked_to_root(solution)
    all_node_within_maximal_distance = instance.all_node_within_maximal_distance(solution)
    is_budget_exceeded = instance.is_budget_not_exceeded(solution)

    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print(f"[INFO] Collected revenue : {revenue}")
    print(f"[INFO] Consummed budget : {consummed} / {instance.B}")
    print(f"[INFO] Sanity check passed : {validity}" +
          f"\n\t No added nodes in solution: {no_added_edges}" +
          f"\n\t All selected nodes are linked to the root : {all_linked_to_root}" + 
          f"\n\t All path are within maximal distance : {all_node_within_maximal_distance}" +
          f"\n\t Budget is not exceeded : {is_budget_exceeded}")
    print("***********************************************************")