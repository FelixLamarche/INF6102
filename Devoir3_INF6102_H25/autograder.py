from pathlib import Path

from utils import Instance

INSTANCES = ["./instances/instance_A_30_10_10.txt", "./instances/instance_B_100_10_10.txt",
             "./instances/instance_C_150_15_10.txt", "./instances/instance_D_150_15_10.txt",
             "./instances/instance_E_200_15_10.txt", "./instances/instance_F_200_15_10.txt"]

LOWER_BOUND = [1471, 10340, 25076, 18098, 16127, 18289]
UPPER_BOUND = [2270, 17309, 38477, 27348, 22233, 24448]
COEF = [1, 1, 1, 1, 2, 2]

if __name__ == '__main__':
    grade = 0
    for instanceName,lb,ub,coef in zip(INSTANCES, LOWER_BOUND, UPPER_BOUND, COEF):
        try:
            inst = Instance(instanceName)
            instanceId = Path(instanceName).stem
            try:
                sol = inst.read_solution(f'./solutions/{instanceId}.txt')
                cost,validity = inst.solution_cost_and_validity(sol)
                if validity:
                    instance_grade = min(coef, round(coef*(1 - (min(cost, ub) - lb) / (ub - lb)), 2))
                    grade += instance_grade
                    print(f'{instanceId} : {instance_grade} / {coef}')
                else:
                    print(f'{instanceId} : 0 / {coef} (file ./solutions/{instanceId}.txt invalid solution)')

            except FileNotFoundError as _:
                print(f'instance{instanceId} : 0 / {coef} (file ./solutions/{instanceId}.txt not found)')
        except FileNotFoundError as _:
            print(f'instance{instanceId} : 0 / {coef} (file ./instances/{instanceId}.txt not found)')
    print(f'Total {grade} / 8')
