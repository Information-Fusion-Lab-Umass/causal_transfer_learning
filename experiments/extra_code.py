w_s = cp.Variable(M , boolean = True)
ones = np.ones(M)
objective_s = np.dot(w_s, ones)
print(objective_s)
constraints_s = []
for i in range(c_0.shape[0]):
    constraints.append(c_0[i] * w >= 1)

for i in range(c_0.shape[0]):
    constraints.append(c_solved[i] * w == 0)
#
for i in range(M):
    constraints.append(w[i] <= 1)

for i in range(M):
    constraints.append(w[i] >= 0)
#
problem_s = cp.Problem(cp.Minimize(objective_s), constraints = constraints_s)
# print(problem)
problem_s.solve()
print(problem_s)
print("status:", problem_s.status)
print("optimal value:", problem_s.value)
print("optimal var:", w_s.value)
#
#
# #
# # # # initialize W
#
# #
#
# #
# #
# # # Objective: c_1^Tw
# # # Constraint: c_0^Tw > 1
# #
