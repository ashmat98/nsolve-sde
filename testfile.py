import numpy as np

from nsdesolve import simulate_2d, add

print(add(3,4))

o0=1.0
g0=10.0
b=+0.01
samples = 100
th=100.0
ka=0.1

print("start")
ret = simulate_2d(
        np.zeros(samples),np.zeros(samples),np.zeros(samples),np.zeros(samples),
    N=5000, skip=10, samples=100,
    dt=0.00001,
    omega0=o0,gamma0=g0,kappa=ka, theta=th,b=b
)
print("finish")

