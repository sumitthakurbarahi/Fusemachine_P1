import numpy as np
from numpy import pi, sin, cos, sqrt, arctan
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_fourbar(r1,r2,r3,r4,r5,beta,xA,yA,alpha,Xd, Yd, rot_num = 10.33,angle_increment = 0.1):
    crank_angles = np.arange(0,rot_num*2*pi,angle_increment) 
    xD , yD = xA + r1 * cos(alpha), yA + r1 * sin(alpha) #point D
    frames = len(crank_angles)
    xB = np.zeros(frames)
    yB = np.zeros(frames)
    xC = np.zeros(frames)
    yC = np.zeros(frames) 
    xM = np.zeros(frames)
    yM = np.zeros(frames)  
    x_trace = []
    y_trace = []


    for i , crank_angle in enumerate(crank_angles):
        theta2 = crank_angle

        l = 2 * r2 * r3 * cos(theta2) - 2 * r1 * r3
        m = 2 * r2 * r3 * sin(theta2)
        n = r1 ** 2 + r2 ** 2 + r3 ** 2 - r4 ** 2 - 2 * r1 * r2 * cos(theta2)

        # theta3 = arctan((m + sqrt(l**2 + m**2 - n**2 )/(l - n)))
        theta3 = 2*arctan((m - sqrt(l**2 + m**2 - n**2 )) / (l - n))

        #point B
        xB[i] = xA + r2 * cos(theta2 + alpha)
        yB[i] = yA + r2 * sin(theta2 + alpha)


        #point C
        xC[i] = xB[i] + r3 * cos(theta3 + alpha)
        yC[i] = yB[i] + r3 * sin(theta3 + alpha)


        #point M
        #coordinates about ground link frame of reference
        xm = r2 * cos(theta2) + r5 * cos(beta + theta3)
        ym = r2 * sin(theta2) + r5 * sin(beta + theta3)
        #about global axis
        xM[i] = xA + xm * cos(alpha) - ym * sin(alpha)
        yM[i] = yA + xm * sin(alpha) + ym * cos(alpha)

        x_trace.append(xM[i])
        y_trace.append(yM[i])

    def init():
        line.set_data([], [])

        return (line,)


    # animation function
    def animate(i):
        x_points = [xA, xB[i],xM[i], xC[i],xB[i],xC[i], xD]
        y_points = [yA, yB[i],yM[i], yC[i],yB[i],yC[i], yD]

        line.set_data(x_points, y_points)
        traced_line.set_data(x_trace[:i], y_trace[:i])
        return line, traced_line


        # set up the figure and subplot
    fig = plt.figure()
    ax = fig.add_subplot(
        111, aspect="equal", autoscale_on=False, xlim=(-160,160), ylim=(-120, 200)
    )

    # add grid lines, title and take out the axis tick labels
    ax.grid(alpha=0.5)
    ax.set_title("Crank and Rocker Motion")
    ax.set_xlabel("X Axis") # Set x-axis label
    ax.set_ylabel("Y Axis")
    (line, ) = ax.plot(
        [], [], "o-", lw=2.5, color="#2b8cbe")
    traced_line, = ax.plot([], [], color='blue',alpha = 0.8)
    # call the animation
    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(xB), interval=40, blit=True, repeat=False
    )
    plt.scatter(Xd,Yd,marker='o',c="black",alpha= 0.8,s=8)

    # show the animation
    plt.show()
