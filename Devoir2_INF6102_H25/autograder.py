from pathlib import Path

from utils import Instance

INSTANCES = ["./instances/instance_A_75_150_13_85_12.txt", "./instances/instance_B_100_200_25_103_12.txt",
             "./instances/instance_C_500_625_83_342_15.txt", "./instances/instance_D_500_1000_125_272_25.txt",
             "./instances/instance_E_500_2500_125_136_5.txt", "./instances/instance_F_500_12500_125_345_15.txt"]

LOWER_BOUND = [412, 945, 2181, 2415, 171, 179]
UPPER_BOUND = [694, 1225, 2969, 3910, 367, 648]
COEF = [1, 1, 1, 1, 2, 2]

if __name__ == '__main__':
    grade = 0
    for instanceName,lb,ub,coef in zip(INSTANCES, LOWER_BOUND, UPPER_BOUND, COEF):
        try:
            inst = Instance(instanceName)
            instanceId = Path(instanceName).stem
            try:
                sol = inst.read_solution(f'./solutions/{instanceId}.txt')
                revenu,validity = inst.solution_value_and_validity(sol)
                if validity:
                    instance_grade = min(coef, round(coef*(max(revenu, lb) - lb) / (ub - lb), 2))
                    grade += instance_grade
                    print(f'{instanceId} : {instance_grade} / {coef}')
                else:
                    print(f'{instanceId} : 0 / {coef} (file ./solutions/{instanceId}.txt invalid solution)')

            except FileNotFoundError as _:
                print(f'instance{instanceId} : 0 / {coef} (file ./solutions/{instanceId}.txt not found)')
        except FileNotFoundError as _:
            print(f'instance{instanceId} : 0 / {coef} (file ./instances/{instanceId}.txt not found)')
    print(f'Total {grade} / 8')
