from utils import Instance, Solution

def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of iterators on grouped node ids (int)
    """

    seq = [-1 for _ in range(instance.J)]
    pos = 0

    for j in range(instance.J):
        for c in range(instance.C):
            if instance.order(j,c):
                seq[pos] = c
                pos += 1

    sol = Solution(seq)

    return sol