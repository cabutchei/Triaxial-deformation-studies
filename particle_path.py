from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import symbols
import numpy as np
from sympy.vector import CoordSys3D
from sympy.tensor.array import derive_by_array
from sympy import Matrix, eye
from collections import deque



x_1, x_2, x_2, X_1, X_2, X_3, t = symbols('x_1 x_2 x_3 X_1 X_2 X_3 t')
x_1 = X_1*(3*t+1)   # equações de movimento
x_2 = t**2*X_3  #
x_3 = X_1*(t+1) #
X = (1,1,1) # coordenada material da partícula de interesse
N = CoordSys3D('N')
X = X[0]*N.i + X[1]*N.j + X[2]*N.k
x = x_1*N.i + x_2*N.j + x_3*N.k
t_size = 20 # tamanho do intervalo de domínio
t_step = 0.01   # tamanho de cada incremento para a animação


def get_matrix_from_vector(v):
    return Matrix([v.coeff(N.i), v.coeff(N.j), v.coeff(N.k)])


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "monospace",
    "font.monospace": 'Computer Modern Typewriter',
    "text.latex.preamble": r"\usepackage{{amsmath}}"
})

ax = plt.axes(projection='3d')
t_values = [i*t_step for i in range(round(t_size/t_step) + 1)]
position_vectors = deque([x.subs(t, t_value).subs(X_1, X.coeff(N.i)).subs(X_2, X.coeff(N.j)).subs(X_3, X.coeff(N.k)) for t_value in t_values])
x_data = []
y_data = []
z_data = []
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)
plot, = ax.plot3D(x_data, y_data, z_data)

def update_data(i):
    if len(position_vectors) == 0:
        return
    current_vector = position_vectors.popleft()
    x_data.append(current_vector.coeff(N.i))
    y_data.append(current_vector.coeff(N.j))
    z_data.append(current_vector.coeff(N.k))
    plot.set_xdata(x_data)
    plot.set_ydata(y_data)
    plot.set_3d_properties(z_data)

def matrix_to_latex(matrix):
    if len(matrix) == 0:
        return
    h = []
    for row in matrix:
        d = "&".join([str(round(entry, 2)) for entry in row])
        h.append(d)
    h = r"\\".join(h)
    h = r"\begin{pmatrix}" + h
    h = h + r"\end{pmatrix}"
    h = "".join(h)
    return h

def vector_to_latex(vector):
    d = "&".join([str(round(entry, 2)) for entry in vector])
    d = r"\begin{pmatrix}" + d
    d = d + r"\end{pmatrix}"
    d = "".join(d)
    return d

u = x-X.subs(t, 0)
u = get_matrix_from_vector(u)
displacement_gradient = u.jacobian([X_1, X_2, X_3])
f = eye(3) + displacement_gradient
f = f.subs(t, 1)
s = f.transpose() * f
s = np.array(s).astype(np.float64)
eigenvalues, eigenvectors = np.linalg.eig(s)
canonical_to_eigvects = np.linalg.inv(eigenvectors)
eigvects_to_canonical = eigenvectors
u_diagonal = np.diag([eigval ** 1/2 for eigval in eigenvalues])
u =  np.matmul(eigvects_to_canonical, np.matmul(u_diagonal, canonical_to_eigvects))
q = np.linalg.inv(u)
v = np.matmul(np.matmul(q, u), np.transpose(q))
eigenvalues = [round(eigval, 2) for eigval in eigenvalues]
eigenvectors = [[round(entry, 2) for entry in eigvect] for eigvect in eigenvectors]
ax.text2D(0, 0.9, f"Valores principais: ${eigenvalues[0]}$, ${eigenvalues[1]}$, ${eigenvalues[2]}$", transform=ax.transAxes)
ax.text2D(0, 1, f"Direções principais: ${vector_to_latex(eigenvectors[0])}$, ${vector_to_latex(eigenvectors[1])}$,\n${vector_to_latex(eigenvectors[1])}$", transform=ax.transAxes)
ax.text2D(0, 0, f"$U = {matrix_to_latex(u)}$", transform=ax.transAxes)
ax.text2D(0.5, 0, f"$Q = {matrix_to_latex(q)}$", transform=ax.transAxes)
ax.quiver(0, 0, 0, eigenvectors[0][0], eigenvectors[0][1], eigenvectors[0][2], color='r', arrow_length_ratio=0.1, length=5)
ax.quiver(0, 0, 0, eigenvectors[1][0], eigenvectors[1][1], eigenvectors[1][2], color='r', arrow_length_ratio=0.1, length=5)
ax.quiver(0, 0, 0, eigenvectors[2][0], eigenvectors[2][1], eigenvectors[2][2], color='r', arrow_length_ratio=0.1, length=5)

animation = FuncAnimation(plt.gcf(), update_data, interval=100, frames=1000, repeat=False)

plt.show()



