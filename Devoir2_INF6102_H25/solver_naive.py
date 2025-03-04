from utils import Instance, Solution

def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of iterators on grouped node ids (int)
    """

    sol = []
    queue = [(0, instance.get_root())]
    cost = 0

    while len(queue) > 0:
        distance, current = queue.pop(0)

        if distance > instance.H:
            continue

        for edge in instance.get_neighbors(current):
            if (cost + edge.cost()) <= instance.B \
                and (distance + 1) <= instance.H:
                sol.append(edge)
                cost += edge.cost()
                for node in edge.idx().difference((current,)):
                    queue.append((distance + 1, node))

    sol = Solution(set(sol))

    return sol