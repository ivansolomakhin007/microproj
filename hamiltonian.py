import numpy as np
from math import cos, sin, pi
from scipy.integrate import ode
import vtk

# массы
m1 = 1
m2 = 1
# длины стержней
l1 = 0.4
l2 = 0.4
# гравитация
g = 9.81


def initial_conditions(p1_0, p2_0, phi1_0, phi2_0):
    """Считаем по начальным условиям интегралы движения (величины, постоянные в каждый
    момент времени жизни системы"""
    C1 = p1_0 * p2_0 * sin(phi1_0 - phi2_0) / (l1 * l2 * (m1 + m2 * sin(phi1_0 - phi2_0) ** 2))
    C2 = (l2 ** 2 * m2 * p1_0 ** 2 + l1 ** 2 * (m1 + m2) * p2_0 ** 2 - l1 * l2 * m2 * p1_0 * p2_0 * cos(
        phi1_0 - phi2_0)) * (sin(2 * (phi1_0 - phi2_0))) / \
         2 * (l1 * l2 * (m1 + m2 * sin(phi1_0 - phi2_0) ** 2)) ** 2
    return C1, C2


# вычисляем C1 и C2
print("Задаем начальные условия")
# p1_0 = float(input("Введите p1: "))
# p2_0 = float(input("Введите p2: "))
p1_0 = 0
p2_0 = 0
# phi1_0 = float(input("Введите phi1: "))
# phi2_0 = float(input("Введите phi2: "))
phi1_0 = pi/3
phi2_0 = pi/3
# задаем начальные условия в вектор
vars0, t0 = np.array([p1_0, p2_0, phi1_0, phi2_0]), 0
C1, C2 = initial_conditions(p1_0, p2_0, phi1_0, phi2_0)


# правая часть уравнений в гамильтоновых переменных
def f(t, variables):
    phi1, phi2, p1, p2 = variables
    d_phi1 = (l2 * p1 - l1 * p2 * cos(phi1 - phi2)) / (l1 ** 2 * l2 * (m1 + m2 * sin(phi1 - phi2) ** 2))
    d_phi2 = (l1 * (m1 + m2) * p2 - l2 * m2 * p1 * cos(phi1 - phi2)) / (
            l1 * l2 ** 2 * (m1 + m2 * sin(phi1 - phi2) ** 2))
    d_p1 = -(m1 + m2) * g * l1 * sin(phi1) - C1 + C2
    d_p2 = -m2 * g * l2 * sin(phi2) + C1 - C2
    return np.array([d_phi1, d_phi2, d_p1, d_p2])



def step_handler(t, vars):
    """Обработчик шага"""
    ts.append(t)
    ys.append(list(vars.copy()))
    y1, y2, y3, y4 = vars
    pass


tmax = 10
ODE = ode(f)
#print("интегрируем")

ODE.set_integrator('dopri5', max_step=0.01)
ODE.set_solout(step_handler)

ts, ys = [], []
ODE.set_initial_value(vars0, t0)  # задание начальных значений
ODE.integrate(tmax)  # решение ОДУ
print(ys)
# print('Flight time = %.4f Distance = %.4f Height =%.4f ' % (FlightTime, Distance, Height))

# визуализация
# распаковываем переменные

def get_xy_coords(p, lengths=np.array([l1, l2])):
    """Get (x, y) coordinates from generalized coordinates p"""
    print(p)
    p = np.atleast_2d(p)
    zeros = np.zeros(p.shape[0])[:, None]
    # x = np.hstack([zeros, lengths * np.sin(p[:, :n])])
    # y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])
    # print(np.cumsum(x, 1))
    phi1 = p[:, 2]
    phi2 = p[:, 3]
    y = np.stack([np.zeros(len(phi1)), -(l1 * np.cos(phi1)), -(l1 * np.cos(phi1) + l2 * np.cos(phi2))], axis=-1)
    x = np.stack([np.zeros(len(phi1)), l1 * np.sin(phi1), l1 * np.sin(phi1) + l2 * np.sin(phi2)], axis=-1)
    print(x**2 + y**2)
    #return np.cumsum(x, 1), np.cumsum(y, 1)
    return x, y

from matplotlib import animation
import matplotlib.pyplot as plt

def animate_pendulum(n):
    x, y = get_xy_coords(ys)
    print(y)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.set(xlim=(-1, 1), ylim=(-1, 1))

    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        line.set_data(x[i], y[i])
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=len(ts),
                                   interval=1000 * max(ts) / len(ts),
                                   blit=True, init_func=init)
    plt.show()
    plt.close(fig)

    return anim


anim = animate_pendulum(3)

