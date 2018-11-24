import numpy as np

a = np.array([1,2]).reshape(-1, 1)
b = np.array([4,5]).reshape(1, -1)

a + b

a = np.array([[1,0],[2,0]]).reshape(-1, 1, 2)
b = np.array([[4,0],[5,0]]).reshape(1, -1, 2)

np.sqrt(((a - b) ** 2).sum(axis=2))

np.linalg.norm(a - b, axis=2)
np.tensordot(a, b, axes=2)

# This seems to work, but why are there the extra dimensions?
np.tensordot(a, b, axes=[2,2])

np.tensordot(a, b, axes=((1, 2), (0, 2)))

# Workaround?
z = np.zeros((a.shape[0], b.shape[1]))

for i in range(a.shape[0]):
    for j in range(b.shape[1]):
        z[i, j] = a[i, 0] @ b[0, j]

z



a = np.array([[1,0],[2,0]]).reshape(-1, 1, 2)
b = np.array([[4,0],[5,0]]).reshape(1, -1, 2)

za = np.zeros_like(a)
zb = np.zeros_like(b)

aa = a + zb
bb = b + za

np.tensordot(aa, bb, axes=((1,2), (0,2)))
# array([[ 8,  0],
#        [16,  0]])


# One loop workaround

a = np.array([[1,0],[2,0]])
b = np.array([[4,0],[5,0]])

bb = b.T

rows = []

for i in range(a.shape[0]):
    rows.append(a[i] @ bb)

res = np.array(rows)

np.array(rows)