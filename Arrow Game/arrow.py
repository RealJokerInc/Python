import numpy as np
import random

def buildGrid(m,n):
    matrix = np.random.randint(1, 5, size=(m, n))
    return matrix

def getCoordinate(a,matrix):
    dim = getSize(matrix)
    coordinate = [int(np.ceil(a/dim[1]))-1,(a+dim[1]-1)%dim[1]]
    return coordinate

def getSize(matrix):
    dim = matrix.shape
    return dim
    
def action(a,matrix):
    coordinate = getCoordinate(a,matrix)
    dim = getSize(matrix)

    border = 0
    ud = 1 #up (0) or down (1)
    lr = 1 #left (0) or right (1)
     
    if coordinate[0] == 0 or coordinate[0] == dim[0]-1:
        border = border + 1

    if coordinate[1] == 0 or coordinate[1] == dim[1]-1:
        border = border + 1
        
    if coordinate[0] == 0:
        if coordinate[1] == 0:
            border = 2
            ud = 0
        elif coordinate[1] == coordinate[0] == dim[0]-1:
            border = 2


    loop = [-1,0,1]
    loop2 = [0,1]

    if border == 0:
        for i in loop:
            for j in loop:
                matrix[coordinate[0]+i,coordinate[1]+j] = rotate(matrix[coordinate[0]+i,coordinate[1]+j])
    
    
    return matrix

def rotate(a):
    if a == 4: 
        return 1
    else:
        return a + 1



