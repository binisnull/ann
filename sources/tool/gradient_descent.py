# -*- coding: UTF-8 -*-
"""
Author: BinIsNull

Contact: dengz004@163.com
"""
# Python 3.8.16

import matplotlib.pyplot as mplt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# surface
num = 2000
wdt = 3
x, y = np.meshgrid(np.linspace(-wdt, wdt, num), np.linspace(-wdt, wdt, num))
z = (x + x ** 2 + y ** 3 + x ** 5) * np.exp(- x ** 2 - y ** 2)
# curve


def grd_des(_x, _y, rate):
    """Gradient Descent"""
    _x -= rate * (1 + 2 * _x + 5 * _x ** 4 - 2 * (_x ** 2 + _x ** 3 + _x * _y ** 3 + _x ** 6)) * np.exp(
            -_x ** 2 - _y ** 2)
    _y -= rate * (3 * _y ** 2 - 2 * (_x * _y + _x ** 2 * _y + _y ** 4 + _x ** 5 * _y)) * np.exp(-_x ** 2 - _y ** 2)
    return _x, _y


stride = 62
inx, iny, rl = 1.05, 0.4, 0.2
x_1 = []
y_1 = []
z_1 = []
for i in range(stride):
    x_1.append(inx)
    y_1.append(iny)
    inz = (inx + inx ** 2 + iny ** 3 + inx ** 5) * np.exp(- inx ** 2 - iny ** 2)
    z_1.append(inz)
    inx, iny = grd_des(inx, iny, rl)
inx, iny = 0.9, -0.2
x_2 = []
y_2 = []
z_2 = []
for i in range(stride):
    x_2.append(inx)
    y_2.append(iny)
    inz = (inx + inx ** 2 + iny ** 3 + inx ** 5) * np.exp(- inx ** 2 - iny ** 2)
    z_2.append(inz)
    inx, iny = grd_des(inx, iny, rl)
#
fig = mplt.figure("3D Surface", facecolor='white', figsize=(7, 7))
ax3d = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax3d)
#
ax3d.view_init(elev=35, azim=175)  # Adjust perspective.
ax3d.set_axis_off()  # Hide axis.
#
# ax3d.plot_surface(x, y, z, cstride=10, rstride=10, cmap='rainbow_r', alpha=0.6)
ax3d.contour3D(x, y, z, 75, cmap='cool', alpha=0.4)
ax3d.plot(x_1, y_1, z_1, color='green', marker='.')
ax3d.plot(x_2, y_2, z_2, color='red', marker='_')
mplt.show()
