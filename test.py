import numpy as np

a=np.array([[0.7,0.2,0.1],
            [0.3,0.5,0.2],
            [0.2,0.2,0.6]])

b=np.argsort(a)[::-1]
print(b)
