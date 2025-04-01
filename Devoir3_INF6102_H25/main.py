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
    print("[INFO] Start the solving: Planing production")
    print("[INFO] input file: %s" % instance.filepath)
    print("[INFO] number of days: %s" % (instance.J))
    print("[INFO] number of products: %s" % (instance.C))
    print("[INFO] storage cost : %s" % (instance.H))
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
    
    cost, validity = instance.solution_cost_and_validity(solution)
    all_asked_products = instance.all_asked_products(solution)
    no_late_deliveries = instance.no_late_deliveries(solution)

    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print(f"[INFO] Solution cost : {cost}")
    print(f"[INFO] Sanity check passed : {validity}" +
          f"\n\t All asked product are delivered : {all_asked_products}" +
          f"\n\t No late deliveries : {no_late_deliveries}")
    print("***********************************************************")