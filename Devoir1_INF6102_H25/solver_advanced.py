from utils import Node, Instance, Solution

class CustomNode(Node):

    """ You are completely free to extend classes defined in utils,
        this might prove useful or enhance code readability
    """

    def __init__(self, idx, neighbors):
        super().__init__(idx, neighbors)

    
def solve(instance: Instance) -> Solution:
    """Write your code here

    Args:
        instance (Instance): An Instance object containing all you need to solve the problem

    Returns:
        Solution: A solution object initialized with 
                  a list of iterators on grouped node ids (int)
    """
    raise Exception('This should be implemented')
