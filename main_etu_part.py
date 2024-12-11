import spatialmath
from lab_tools_etu import matrix_interaction
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.backends.PyPlot import PyPlot
import matplotlib
import math
from matplotlib import pyplot

matplotlib.use('QT5Agg')
''' Sampling period '''
Te = 5e-2
duration = 100
DRAW = True

''' Intrinsic Camera parameters '''
f = 8e-3
kx = 10e-6
ky = 10e-6

''' 'Figure image' '''
pyplot.ion()

u1 = -421.03
v1 = 421.08
u2 = 421.03
v2 = 421.08
u3 = 421.03
v3 = -420.97
u4 = -421.03
v4 = -420.97

# *****************************************************
#           Début de zone de modification             
# *****************************************************%

''' Positions désirées '''
''' Il faut décommenter les positions désirés en suivant les indications du sujet. '''


''' first desired image points '''

'''u1_des = -421.03 - 400
v1_des = 421.08
u2_des = 421.03 - 400
v2_des = 421.08
u3_des = 421.03 - 400
v3_des = -420.97
u4_des = -421.03 - 400
v4_des = -420.97 '''

''' second desired image points '''
'''
u1_des = -421.03
v1_des = 421.08 - 400
u2_des = 421.03
v2_des = 421.08 - 400
u3_des = 421.03
v3_des = -420.97 - 400
u4_des = -421.03
v4_des = -420.97 - 400'''

''' third desired image points '''
'''
u1_des = -421.03 + 300
v1_des = 421.08 - 400
u2_des = 421.03 + 300
v2_des = 421.08 - 400
u3_des = 421.03 + 300
v3_des = -420.97 - 400
u4_des = -421.03 + 300
v4_des = -420.97 - 400'''

''' A more complex motion'''
'''
u1_des = -580
v1_des = -17
u2_des = -50
v2_des = -110
u3_des = -158
v3_des = -740
u4_des = -847
v4_des = -667'''

''' 0.2 rad rotation in the image plane'''
'''
u1_des = -328.98
v1_des = 496.34
u2_des = 496.29
v2_des = 329.04
u3_des = 329
v3_des = -496.23
u4_des = -496.27
v4_des = -328.93'''

''' pi/2 rad rotation in the image plane'''
'''
u1_des = 421.08
v1_des = 421.03
u2_des = 421.08
v2_des = -421.03
u3_des = -420.97
v3_des = -421.03
u4_des = -420.97
v4_des = 421.03 '''

'''pi rotation in the image plane'''

u1_des = 421.03
v1_des = -421.08
u2_des = -421.03
v2_des = -421.08
u3_des = -421.03
v3_des = 420.97
u4_des = 421.03
v4_des = 420.97

# *****************************************************
#           Fin de zone de modification             
# *****************************************************%

# s_des = np.transpose(np.array([u1_des, v1_des, u2_des, v2_des, u3_des, v3_des, u4_des, v4_des]))
s_des = np.array([[u1_des], [v1_des], [u2_des], [v2_des], [u3_des], [v3_des], [u4_des], [v4_des]])

fig, ax = pyplot.subplots()
mngr = pyplot.get_current_fig_manager()
mngr.window.setGeometry(1000,300,600,600)
pyplot.xlim([-1000.0, 1000.0])
pyplot.ylim([-1000.0, 1000.0])
pyplot.xlabel('pixels')
pyplot.ylabel('pixels')
pyplot.title('Camera Image')
ax.set_aspect('equal')

''' Current points in the camera image'''
point1, = ax.plot(u1, v1, 'r', marker='x', markersize=5)
point2, = ax.plot(u2, v2, 'k', marker='x', markersize=5)
point3, = ax.plot(u3, v3, 'b', marker='x', markersize=5)
point4, = ax.plot(u4, v4, 'm', marker='x', markersize=5)

''' Desired points in the camera image'''
point1_des, = ax.plot(u1_des, v1_des, 'r', marker='o', markersize=5)
point2_des, = ax.plot(u2_des, v2_des, 'k', marker='o', markersize=5)
point3_des, = ax.plot(u3_des, v3_des, 'b', marker='o', markersize=5)
point4_des, = ax.plot(u4_des, v4_des, 'm', marker='o', markersize=5)

''' Create robot and World points '''
puma = rtb.models.Puma560()
P0 = np.zeros((4, 4))
P0[0, :] = np.array([6.10861770e-01 + 0.3, -0.1295 + 0.3, 0, 1.0])
P0[1, :] = np.array([6.10861770e-01 - 0.3, -0.1295 + 0.3, 0, 1.0])
P0[2, :] = np.array([6.10861770e-01 - 0.3, -0.1295 - 0.3, 0, 1.0])
P0[3, :] = np.array([6.10861770e-01 + 0.3, -0.1295 - 0.3, 0, 1.0])

''' Launch backend, display Robot and World points'''
backend = PyPlot()
backend.launch(name='robot', limits=[-0.9, 0.9, -0.7, 0.7, 0, 1.0])
backend.add(puma)
Q = np.array([0.0, math.pi / 4, -math.pi, 0, -math.pi / 4, 0.0])
puma.q = np.transpose(Q)
# spatialmath.baseposematrix.base.plot_point([0.4, 0.4, 0],'r*')#,marker="bs")
spatialmath.base.plot_point(P0[0, 0:3], 'r*')
spatialmath.base.plot_point(P0[1, 0:3], 'k*')
spatialmath.base.plot_point(P0[2, 0:3], 'b*')
spatialmath.base.plot_point(P0[3, 0:3], 'm*')

# *****************************************************
#           Début de zone de modification             
# *****************************************************%

''' Compléter matrice des paramètres intrinsèques pour un point '''

# a1 = ...

''' Compléter matrice des paramètres intrinsèques pour 4 points '''
# A1 = ...

# *****************************************************
#           Fin de zone de modification             
# *****************************************************%

'''Area and angle computation'''
a1_des = np.array([[u2_des - u1_des, v2_des - v1_des, 1]])
a2_des = np.array([[u4_des - u3_des, v4_des - v3_des, 1]])

area_des = np.sqrt(np.linalg.norm(np.cross(a1_des, a2_des)))

vec1_des = np.array([u2_des - u1_des, v2_des - v1_des, 0.0])
vec2_des = np.array([5.0, 0.0, 0.0])

angle_des = np.arctan2(np.linalg.norm(np.cross(vec1_des, vec2_des)), np.dot(vec1_des, vec2_des))

'''Storage preparation'''

Mat_init = puma.fkine(Q)
X_store = Mat_init.A[0, 3]
Y_store = Mat_init.A[1, 3]
Z_store = Mat_init.A[2, 3]
control_vx_store = 0.0
control_vy_store = 0.0
control_vz_store = 0.0
control_wx_store = 0.0
control_wy_store = 0.0
control_wz_store = 0.0
area_store = area_des
area_des_store = area_des
angle_store = 0.0
u1_store = u1
v1_store = v1
u2_store = u2
v2_store = v2
u3_store = u3
v3_store = v3
u4_store = u4
v4_store = v4
time_store = 0.0


for i in range(duration):

    ''' Compute image points '''

    ''' Homogenous Transform from the camera to World frame'''
    M0c = puma.fkine(puma.q)
    Mc0 = np.linalg.inv(M0c.A)

    ''' Pc : World points expressed in the camera frame (m), (X_i, Y_i and Z_i) in the lecutre'''
    P01 = np.transpose(P0)
    Pc = Mc0 @ P01

    ''' pc : Image of World points in m (x and y in rows) (x_i and yi in the lecture)'''
    pc = np.zeros((4, 4))
    # print(Pc)
    for j in range(4):
        pc[0, j] = Pc[0, j] / Pc[2, j]
        pc[1, j] = Pc[1, j] / Pc[2, j]
        
    '''u_i, v_i : Image points in pixels '''

    u1 = (pc[0, 0] * f) / kx
    v1 = (pc[1, 0] * f) / ky

    u2 = (pc[0, 1] * f) / kx
    v2 = (pc[1, 1] * f) / ky

    u3 = (pc[0, 2] * f) / kx
    v3 = (pc[1, 2] * f) / ky

    u4 = (pc[0, 3] * f) / kx
    v4 = (pc[1, 3] * f) / ky

    s = np.array([[u1], [v1], [u2], [v2], [u3], [v3], [u4], [v4]])

    ''' Control law implementation '''

    # *****************************************************
    #           Début de zone de modification             
    # *****************************************************%

    ''' Image jacobian '''

    ''' Compléter en utilisant matrix_interaction / lab_tools_etu '''
    # Ls = ...

    # *****************************************************
    #           Fin de zone de modification             
    # *****************************************************%


    '''Area and angle computation'''
    a1 = np.array([[u2 - u1, v2 - v1, 1]])
    a2 = np.array([[u4 - u3, v4 - v3, 1]])

    area = np.sqrt(np.linalg.norm(np.cross(a1, a2)))

    vec1 = np.array([u2 - u1, v2 - v1, 0.0])
    vec2 = np.array([5.0, 0.0, 0.0])

    angle = np.arctan2(np.linalg.norm(np.cross(vec1, vec2)), np.dot(vec1, vec2))

    if angle < 0:
        angle += 2*math.pi
    if angle > 2*math.pi:
        angle += -2*math.pi


    # *****************************************************
    #           Début de zone de modification             
    # *****************************************************%

    ''' control law independent form the robot'''

    '''Compléter '''
    ''' J_xy de l'article '''
    # Lxy = ...

    ''' J_z de l'article '''
    # Lz = ...


    ''''Compléter'''
    ''' control part 1 : vz, wz '''
    # gamma_omega_z = ...
    # gamma_t_z = ...

    # control_1 = ...

    '''control part 2: vx, vy, wx, wy'''
    # control_2 = ...

    ''''Modifier'''
    ''' control law including the robot '''
    control = np.zeros([6,1])


    # *****************************************************
    #           Fin de zone de modification             
    # *****************************************************%

    ''''robot difference kinematic simulation '''
    for j in range(6):
        Q[j] += control[j] * Te

    puma.q = Q
    backend.step(Te)
    if i % 2 == 0:
        x = 0
        y = 0
    else:
        x = 1
        y = 1
    point1.set_ydata(v1)
    point1.set_xdata(u1)
    point2.set_ydata(v2)
    point2.set_xdata(u2)
    point3.set_ydata(v3)
    point3.set_xdata(u3)
    point4.set_ydata(v4)
    point4.set_xdata(u4)

    '''storage'''
    angle_store = np.r_[angle_store, angle]
    area_store = np.r_[area_store, area]
    X_store = np.r_[X_store, M0c.A[0, 3]]
    Y_store = np.r_[Y_store, M0c.A[1, 3]]
    Z_store = np.r_[Z_store, M0c.A[2, 3]]
    control_vx_store = np.r_[control_vx_store, control[0]]
    control_vy_store = np.r_[control_vy_store, control[1]]
    control_vz_store = np.r_[control_vz_store, control[2]]
    control_wx_store = np.r_[control_wx_store, control[3]]
    control_wy_store = np.r_[control_wy_store, control[4]]
    control_wz_store = np.r_[control_wz_store, control[5]]
    u1_store = np.r_[u1_store, u1]
    v1_store = np.r_[v1_store, v1]
    u2_store = np.r_[u2_store, u2]
    v2_store = np.r_[v2_store, v2]
    u3_store = np.r_[u3_store, u3]
    v3_store = np.r_[v3_store, v3]
    u4_store = np.r_[u4_store, u4]
    v4_store = np.r_[v4_store, v4]
    time_store = np.r_[time_store, i*Te]
#
fig.canvas.draw()
#

if DRAW:

    fig1, ax1 = pyplot.subplots()
    ax1.plot(time_store, u1_store, 'r-', linewidth=1.5, label='u1')
    ax1.plot(time_store, u2_store, 'k-', label='u2')
    ax1.plot(time_store, u3_store, 'b-', label='u3')
    ax1.plot(time_store, u4_store, 'm-', label='u4')
    ax1.set_facecolor('w')

    ax1.plot(time_store, u1_des*np.ones((time_store.shape[0], 1)), 'r-*', linewidth=1.5, label='u1_des')
    ax1.plot(time_store, u2_des*np.ones((time_store.shape[0], 1)), 'k--', linewidth=1.5, label='u2_des')
    ax1.plot(time_store, u3_des*np.ones((time_store.shape[0], 1)), 'b-*', linewidth=1.5, label='u3_des')
    ax1.plot(time_store, u4_des*np.ones((time_store.shape[0], 1)), 'm--', linewidth=1.5, label='u4_des')
    pyplot.legend(facecolor='white', framealpha=0)
    pyplot.xlabel('time(seconds)')
    pyplot.ylabel('pixels')
    fig1.suptitle('image points position along the x-axis', fontsize=13)


    fig2, ax2 = pyplot.subplots()
    ax2.plot(time_store, v1_store, 'r-', linewidth=1.5, label='v1')
    ax2.plot(time_store, v2_store, 'k-', label='v2')
    ax2.plot(time_store, v3_store, 'b-', label='v3')
    ax2.plot(time_store, v3_store, 'm-', label='v3')
    ax2.set_facecolor('w')

    ax2.plot(time_store, v1_des*np.ones((time_store.shape[0], 1)), 'r-*', linewidth=1.5, label='v1_des')
    ax2.plot(time_store, v2_des*np.ones((time_store.shape[0], 1)), 'k--', linewidth=1.5, label='v2_des')
    ax2.plot(time_store, v3_des*np.ones((time_store.shape[0], 1)), 'b-*', linewidth=1.5, label='v3_des')
    ax2.plot(time_store, v4_des*np.ones((time_store.shape[0], 1)), 'm--', linewidth=1.5, label='v4_des')
    pyplot.legend(facecolor='white', framealpha=0)
    pyplot.xlabel('time(seconds)')
    pyplot.ylabel('pixels')
    fig2.suptitle('image points position along the y-axis', fontsize=13)

    fig3, ax3 = pyplot.subplots()
    ax3.plot(u1_store, v1_store, 'r-', linewidth=1, label='pt1')
    ax3.plot(u2_store, v2_store, 'k--', linewidth=1, label='pt2')
    ax3.plot(u3_store, v3_store, 'b-', linewidth=1, label='pt3')
    ax3.plot(u4_store, v4_store, 'm-', linewidth=1, label='pt4')

    ax3.plot(u1_store[0], v1_store[0], 'rx')
    ax3.plot(u1_des, v1_des, 'ro')
    ax3.plot(u2_store[0], v2_store[0], 'kx')
    ax3.plot(u2_des, v2_des, 'ko')
    ax3.plot(u3_store[0], v3_store[0], 'bx')
    ax3.plot(u3_des, v3_des, 'bo')
    ax3.plot(u4_store[0], v4_store[0], 'mx')
    ax3.plot(u4_des, v4_des, 'mo')
    ax3.set_aspect('equal')
    ax3.set_facecolor('w')
    fig3.suptitle('Image Trajectory', fontsize=13)
    pyplot.xlabel('pixels')
    pyplot.ylabel('pixels')

    fig4, ax4 = pyplot.subplots()
    ax4.plot(time_store, angle_store, 'r-', linewidth=1.5, label='angle')
    ax4.plot(time_store, angle_des*np.ones((time_store.shape[0], 1)), 'r--', linewidth=1.5, label='angle des')
    pyplot.xlabel('time (seconds)')
    pyplot.ylabel('angle (rad)')
    ax4.set_facecolor('w')
    pyplot.legend(facecolor='white', framealpha=0)
    fig4.suptitle('angle trajectory', fontsize=13)

    fig5, ax5 = pyplot.subplots()
    ax5.plot(time_store, area_store, 'r-', linewidth=1.5, label='area')
    ax5.plot(time_store, area_des*np.ones((time_store.shape[0], 1)), 'r--', linewidth=1.5, label='area des')
    pyplot.xlabel('time (seconds)')
    pyplot.ylabel('area (pixels*pixels)')
    ax5.set_facecolor('w')
    pyplot.legend(facecolor='white', framealpha=0)
    fig5.suptitle('area trajectory', fontsize=13)

    fig6, ax6 = pyplot.subplots()
    ax6.plot(time_store, X_store, 'r-', linewidth=1.5, label='X axis')
    ax6.plot(time_store, Y_store, 'g-', linewidth=1.5, label='Y axis')
    ax6.plot(time_store, Z_store, 'b-', linewidth=1.5, label='Z axis')
    pyplot.xlabel('time (seconds)')
    pyplot.ylabel('m')
    ax6.set_facecolor('w')
    pyplot.legend(facecolor='white', framealpha=0)
    fig6.suptitle('Camera frame origin position in the robot base frame', fontsize=13)

    fig7, ax7 = pyplot.subplots()
    ax7.plot(time_store, control_vx_store, 'r-', linewidth=1.5, label='X axis')
    ax7.plot(time_store, control_vy_store, 'g-', linewidth=1.5, label='Y axis')
    ax7.plot(time_store, control_vz_store, 'b-', linewidth=1.5, label='Z axis')
    pyplot.xlabel('time (seconds)')
    pyplot.ylabel('m/s')
    ax7.set_facecolor('w')
    pyplot.legend(facecolor='white', framealpha=0)
    fig7.suptitle('Operational end-frame origin of the robot in the robot end-effector frame', fontsize=13)

    fig8, ax8 = pyplot.subplots()
    ax8.plot(time_store, control_wx_store, 'r-', linewidth=1.5, label='X axis')
    ax8.plot(time_store, control_wy_store, 'g-', linewidth=1.5, label='Y axis')
    ax8.plot(time_store, control_wz_store, 'b-', linewidth=1.5, label='Z axis')
    pyplot.xlabel('time (seconds)')
    pyplot.ylabel('rad/s')
    ax8.set_facecolor('w')
    pyplot.legend(facecolor='white', framealpha=0)
    fig8.suptitle('Operational end-frame rotation veloctiy of the robot in the robot end-effector frame', fontsize=13)

backend.hold()
