from numpy import fromfile

t = fromfile("solution_dual_t.data", sep=" ")
u = fromfile("solution_dual_u.data", sep=" ")
k = fromfile("solution_dual_k.data", sep=" ")
r = fromfile("solution_dual_r.data", sep=" ")

u.shape = len(u)//3, 3
k.shape = len(k)//3, 3
r.shape = len(r)//3, 3

