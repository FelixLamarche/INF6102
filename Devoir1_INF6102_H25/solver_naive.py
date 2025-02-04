from utils import Instance, Solution

def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of iterators on grouped node ids (int)
    """
    return Solution([[node.get_idx()] for node in instance.nodes])
