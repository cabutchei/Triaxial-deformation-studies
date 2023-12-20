from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import symbols
from sympy.vector import CoordSys3D
from sympy.tensor.array import derive_by_array
from sympy import Matrix
x_1, x_2, x_2, X_1, X_2, X_3, t = symbols('x_1 x_2 x_3 X_1 X_2 X_3 t')
x_1 = X_1*(3*t+1)
x_2 = t**2*X_3
x_3 = X_1*(t+1)
X = (1,1,1)
N = CoordSys3D('N')
x = x_1*N.i + x_2*N.j + x_3*N.k
print(x_1.coeff(X_2))

ax = plt.axes(projection='3d')
t_size = 5
step_size = 0.1
t_values = [step_size*i for i in range(t_size * 10 + 1)]
position_vectors = [x.subs(t, t_value).subs(X_1, X[0]).subs(X_2, X[1]).subs(X_3, X[2]) for t_value in t_values]
x_data = []
y_data = []
z_data = []
for vector in position_vectors:
    x_data.append(vector.coeff(N.i))
    y_data.append(vector.coeff(N.j))
    z_data.append(vector.coeff(N.k))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_data, y_data, z_data)



