import math
import numpy as np

def exact_vmf(kappa, mu, x):
    C = kappa / (4 * math.pi * math.sinh(kappa))
    dot = np.dot(mu, x)
    return C * math.exp(kappa * dot)

print(exact_vmf(10.0, [1,0,0], [1,0,0]))
