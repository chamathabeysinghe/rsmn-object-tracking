from pulp import *

# my_lp_problem = LpProblem("Simp", LpMaximize)
my_lp_problem = LpProblem("My LP Problem", LpMaximize)

x = LpVariable('x', lowBound=0, cat='Continuous')
y = LpVariable('y', lowBound=2, cat='Continuous')

my_lp_problem += 4 * x + 3 * y, "Z"

my_lp_problem += 2 * y <= 25 - x
my_lp_problem += 4 * y >= 2 * x - 8
my_lp_problem += y <= 2 * x - 5


my_lp_problem.solve()
pulp.LpStatus[my_lp_problem.status]



