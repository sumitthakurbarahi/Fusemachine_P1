import numpy as np
import pandas as pd
from numpy import pi,argmin
from scipy.optimize import differential_evolution
from shapely.geometry import Point, Polygon
from fourbar_animator import *


def CPF(x, y):
    '''
    Circular proximity function, checks the closeness of set of points traced by point C of linkage
    with a circular arc. The point which traces arc closest to a circle is takes as point C
    its radius as length of link 4 
    and its centre as fixed pivot of link 4
    '''
    n = len(x)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    
    for i in range(n):
        a[i] = (2/n) * np.sum(x) - 2 * x[i]
        b[i] = (2/n) * np.sum(y) - 2 * y[i]
        c[i] = x[i]**2 + y[i]**2 - (1/n) * np.sum(x**2 + y**2)
    
    f1 = np.sum(a**2)
    f2 = np.sum(b**2)
    f3 = np.sum(2 * a * c)
    f4 = np.sum(2 * b * c)
    f5 = np.sum(2 * a * b)
    
    C = np.linalg.inv(np.array([[2*f1, f5], [f5, 2*f2]])) @ np.array([-f3, -f4])
    Cx, Cy = C[0], C[1]
    
    R2 = np.zeros(n)
    for i in range(n):
        R2[i] = (x[i] - Cx)**2 + (y[i] - Cy)**2
    
    error = np.sum(((R2 - np.mean(R2)) / np.mean(R2))**2)
    R = np.mean(np.sqrt(R2))
    circle = [Cx, Cy, R]
    
    return error, circle

def objfun(X, Xd, Yd):
    xA, yA, r3, beta = X[0], X[1], X[2], X[3]
    n = len(Xd)
    R = np.zeros(n)
    
    for i in range(n):
        R[i] = np.sqrt((Xd[i] - xA)**2 + (Yd[i] - yA)**2)
    
    max_index = np.argmax(R)
    min_index = np.argmin(R)
    
    if Point(xA, yA).within(Polygon(zip(Xd, Yd))):
        r2 = (np.max(R) + np.min(R)) / 2
        r5 = (np.max(R) - np.min(R)) / 2

    else:
        r2 = (np.max(R) - np.min(R)) / 2
        r5 = (np.max(R) + np.min(R)) / 2
    
    xC1, yC1 = np.zeros(n), np.zeros(n)
    xC2, yC2 = np.zeros(n), np.zeros(n)
    
    for i in range(n):
        if ((i < max_index and i > min_index) or (i > max_index and i < min_index)):
            theta2m1 = np.arctan2(Yd[i] - yA, Xd[i] - xA) + np.arccos((r2**2 + R[i]**2 - r5**2) / (2 * r2 * R[i]))
            theta5m1 = np.arctan2(Yd[i] - yA - r2 * np.sin(theta2m1), Xd[i] - xA - r2 * np.cos(theta2m1))
            theta3m1 = theta5m1 - beta
            xC1[i] = xA + r2 * np.cos(theta2m1) + r3 * np.cos(theta3m1)
            yC1[i] = yA + r2 * np.sin(theta2m1) + r3 * np.sin(theta3m1)
            
            theta2m2 = np.arctan2(Yd[i] - yA, Xd[i] - xA) - np.arccos((r2**2 + R[i]**2 - r5**2) / (2 * r2 * R[i]))
            theta5m2 = np.arctan2(Yd[i] - yA - r2 * np.sin(theta2m2), Xd[i] - xA - r2 * np.cos(theta2m2))
            theta3m2 = theta5m2 - beta
            xC2[i] = xA + r2 * np.cos(theta2m2) + r3 * np.cos(theta3m2)
            yC2[i] = yA + r2 * np.sin(theta2m2) + r3 * np.sin(theta3m2)
        elif (i == max_index):
            theta2 = np.arctan2(Yd[i] - yA, Xd[i] - xA)
            theta5 = theta2
            theta3 = theta5 - beta
            xC1[i] = xA + r2 * np.cos(theta2) + r3 * np.cos(theta3)
            yC1[i] = yA + r2 * np.sin(theta2) + r3 * np.sin(theta3)
            xC2[i] = xC1[i]
            yC2[i] = yC1[i]
        elif (i == min_index):
            if Point(xA, yA).within(Polygon(zip(Xd, Yd))):
                theta2 = np.arctan2(Yd[i] - yA, Xd[i] - xA)
                theta5 = np.pi + theta2
                theta3 = theta5 - beta
                xC1[i] = xA + r2 * np.cos(theta2) + r3 * np.cos(theta3)
                yC1[i] = yA + r2 * np.sin(theta2) + r3 * np.sin(theta3)
                xC2[i] = xC1[i]
                yC2[i] = yC1[i]
            else:
                theta2 = np.pi + np.arctan2(Yd[i] - yA, Xd[i] - xA)
                theta5 = np.arctan2(Yd[i] - yA, Xd[i] - xA)
                theta3 = theta5 - beta
                xC1[i] = xA + r2 * np.cos(theta2) + r3 * np.cos(theta3)
                yC1[i] = yA + r2 * np.sin(theta2) + r3 * np.sin(theta3)
                xC2[i] = xC1[i]
                yC2[i] = yC1[i]
        else:
            theta2m1 = np.arctan2(Yd[i] - yA, Xd[i] - xA) + np.arccos((r2**2 + R[i]**2 - r5**2) / (2 * r2 * R[i]))
            theta5m1 = np.arctan2(Yd[i] - yA - r2 * np.sin(theta2m1), Xd[i] - xA - r2 * np.cos(theta2m1))
            theta3m1 = theta5m1 - beta
            xC2[i] = xA + r2 * np.cos(theta2m1) + r3 * np.cos(theta3m1)
            yC2[i] = yA + r2 * np.sin(theta2m1) + r3 * np.sin(theta3m1)
            
            theta2m2 = np.arctan2(Yd[i] - yA, Xd[i] - xA) - np.arccos((r2**2 + R[i]**2 - r5**2) / (2 * r2 * R[i]))
            theta5m2 = np.arctan2(Yd[i] - yA - r2 * np.sin(theta2m2), Xd[i] - xA - r2 * np.cos(theta2m2))
            theta3m2 = theta5m2 - beta
            xC1[i] = xA + r2 * np.cos(theta2m2) + r3 * np.cos(theta3m2)
            yC1[i] = yA + r2 * np.sin(theta2m2) + r3 * np.sin(theta3m2)
    
    C1CPF, circle1 = CPF(xC1, yC1)
    C2CPF, circle2 = CPF(xC2, yC2)
    
    error, index = min([C1CPF, C2CPF]), np.argmin([C1CPF, C2CPF])
    if index == 0:
        xD, yD, r1, r4 = circle1[0], circle1[1], np.sqrt((xA - circle1[0])**2 + (yA - circle1[1])**2), circle1[2]
        alpha = np.arctan2(yD - yA, xD - xA)
    else:
        xD, yD, r1, r4 = circle2[0], circle2[1], np.sqrt((xA - circle2[0])**2 + (yA - circle2[1])**2), circle2[2]
        alpha = np.arctan2(yD - yA, xD - xA)
    
    linkage = [r1, r2, r3, r4, r5, beta, xA, yA, alpha]
    s = np.min(linkage[:4])
    l = np.max(linkage[:4])
    pq = np.sum(linkage[:4]) - (s + l)
    error += (s + l >= pq) * 1000 + (linkage[1] != s) * 1000
    
    return error



def linkOutput(X, Xd, Yd):
    xA, yA, r3, beta = X[0], X[1], X[2], X[3]
    n = len(Xd)
    R = np.zeros(n)
    
    for i in range(n):
        R[i] = np.sqrt((Xd[i] - xA)**2 + (Yd[i] - yA)**2)
    
    max_index = np.argmax(R)
    min_index = np.argmin(R)
    
    if Point(xA, yA).within(Polygon(zip(Xd, Yd))):
        r2 = (np.max(R) + np.min(R)) / 2
        r5 = (np.max(R) - np.min(R)) / 2

    else:
        r2 = (np.max(R) - np.min(R)) / 2
        r5 = (np.max(R) + np.min(R)) / 2
    
    xC1, yC1 = np.zeros(n), np.zeros(n)
    xC2, yC2 = np.zeros(n), np.zeros(n)
    
    for i in range(n):
        if ((i < max_index and i > min_index) or (i > max_index and i < min_index)):
            theta2m1 = np.arctan2(Yd[i] - yA, Xd[i] - xA) + np.arccos((r2**2 + R[i]**2 - r5**2) / (2 * r2 * R[i]))
            theta5m1 = np.arctan2(Yd[i] - yA - r2 * np.sin(theta2m1), Xd[i] - xA - r2 * np.cos(theta2m1))
            theta3m1 = theta5m1 - beta
            xC1[i] = xA + r2 * np.cos(theta2m1) + r3 * np.cos(theta3m1)
            yC1[i] = yA + r2 * np.sin(theta2m1) + r3 * np.sin(theta3m1)
            
            theta2m2 = np.arctan2(Yd[i] - yA, Xd[i] - xA) - np.arccos((r2**2 + R[i]**2 - r5**2) / (2 * r2 * R[i]))
            theta5m2 = np.arctan2(Yd[i] - yA - r2 * np.sin(theta2m2), Xd[i] - xA - r2 * np.cos(theta2m2))
            theta3m2 = theta5m2 - beta
            xC2[i] = xA + r2 * np.cos(theta2m2) + r3 * np.cos(theta3m2)
            yC2[i] = yA + r2 * np.sin(theta2m2) + r3 * np.sin(theta3m2)
        elif (i == max_index):
            theta2 = np.arctan2(Yd[i] - yA, Xd[i] - xA)
            theta5 = theta2
            theta3 = theta5 - beta
            xC1[i] = xA + r2 * np.cos(theta2) + r3 * np.cos(theta3)
            yC1[i] = yA + r2 * np.sin(theta2) + r3 * np.sin(theta3)
            xC2[i] = xC1[i]
            yC2[i] = yC1[i]
        elif (i == min_index):
            if Point(xA, yA).within(Polygon(zip(Xd, Yd))):
                theta2 = np.arctan2(Yd[i] - yA, Xd[i] - xA)
                theta5 = np.pi + theta2
                theta3 = theta5 - beta
                xC1[i] = xA + r2 * np.cos(theta2) + r3 * np.cos(theta3)
                yC1[i] = yA + r2 * np.sin(theta2) + r3 * np.sin(theta3)
                xC2[i] = xC1[i]
                yC2[i] = yC1[i]
            else:
                theta2 = np.pi + np.arctan2(Yd[i] - yA, Xd[i] - xA)
                theta5 = np.arctan2(Yd[i] - yA, Xd[i] - xA)
                theta3 = theta5 - beta
                xC1[i] = xA + r2 * np.cos(theta2) + r3 * np.cos(theta3)
                yC1[i] = yA + r2 * np.sin(theta2) + r3 * np.sin(theta3)
                xC2[i] = xC1[i]
                yC2[i] = yC1[i]
        else:
            theta2m1 = np.arctan2(Yd[i] - yA, Xd[i] - xA) + np.arccos((r2**2 + R[i]**2 - r5**2) / (2 * r2 * R[i]))
            theta5m1 = np.arctan2(Yd[i] - yA - r2 * np.sin(theta2m1), Xd[i] - xA - r2 * np.cos(theta2m1))
            theta3m1 = theta5m1 - beta
            xC2[i] = xA + r2 * np.cos(theta2m1) + r3 * np.cos(theta3m1)
            yC2[i] = yA + r2 * np.sin(theta2m1) + r3 * np.sin(theta3m1)
            
            theta2m2 = np.arctan2(Yd[i] - yA, Xd[i] - xA) - np.arccos((r2**2 + R[i]**2 - r5**2) / (2 * r2 * R[i]))
            theta5m2 = np.arctan2(Yd[i] - yA - r2 * np.sin(theta2m2), Xd[i] - xA - r2 * np.cos(theta2m2))
            theta3m2 = theta5m2 - beta
            xC1[i] = xA + r2 * np.cos(theta2m2) + r3 * np.cos(theta3m2)
            yC1[i] = yA + r2 * np.sin(theta2m2) + r3 * np.sin(theta3m2)
    
    C1CPF, circle1 = CPF(xC1, yC1)
    C2CPF, circle2 = CPF(xC2, yC2)
    
    error, index = min([C1CPF, C2CPF]), np.argmin([C1CPF, C2CPF])
    if index == 0:
        xD, yD, r1, r4 = circle1[0], circle1[1], np.sqrt((xA - circle1[0])**2 + (yA - circle1[1])**2), circle1[2]
        alpha = np.arctan2(yD - yA, xD - xA)
    else:
        xD, yD, r1, r4 = circle2[0], circle2[1], np.sqrt((xA - circle2[0])**2 + (yA - circle2[1])**2), circle2[2]
        alpha = np.arctan2(yD - yA, xD - xA)
    
    linkage = [r1, r2, r3, r4, r5, beta, xA, yA, alpha]
    return linkage

df = pd.read_csv('curve.csv')
Xd = df['x'].values
Yd = df['y'].values
for i in range(len(Xd)):
    Xd[i] = int(Xd[i])
    Yd[i] = int(Yd[i])

boundss = [(-120,120),(-120,120), (1,200), (-pi,pi)]

number_of_runs = 1

# result = differential_evolution(objfun, bounds= boundss, args=(Xd, Yd))

error = np.zeros(number_of_runs)
min_index = 0
output_links = []

for n in range(number_of_runs):
    result = differential_evolution(objfun, bounds= boundss, args=(Xd, Yd))
    error[n]= result.fun
    output_links.append(linkOutput(result.x, Xd, Yd))
    min_index = argmin(error)


r1 = output_links[min_index][0]
r2 = output_links[min_index][1]
r3 = output_links[min_index][2]
r4 = output_links[min_index][3]
r5 = output_links[min_index][4]
beta = output_links[min_index][5]
xA = output_links[min_index][6]
yA= output_links[min_index][7]
alpha = output_links[min_index][8]

animate_fourbar(r1,r2,r3,r4,r5,beta,xA,yA,alpha,Xd,Yd)

#plot the histogram of error values over number_of_runs of program execution
plt.hist(error, bins=10, color='skyblue', edgecolor='black') 
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.grid(True)

plt.show()


#things to add
#automatic scale adjuster
