import numpy as np
from math import cos, sin
from scipy.integrate import ode

# массы
m1 = 0.1
m2 = 0.1
# длины стержней
l1 = 1
l2 = 1
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
p1_0 = float(input("Введите p1: "))
p2_0 = float(input("Введите p2: "))
phi1_0 = float(input("Введите phi1: "))
phi2_0 = float(input("Введите phi2: "))
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


FlightTime, Distance, Height = 0, 0, 0
y4old = 0


def step_handler(t, vars):
    """Обработчик шага"""
    global FlightTime, Distance, Height, y4old
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
print('Flight time = %.4f Distance = %.4f Height =%.4f ' % (FlightTime, Distance, Height))


#TODO пишем визцализацию, видимо будем юзать VTK + PARAVIEW хотя возможно и PYGAME

