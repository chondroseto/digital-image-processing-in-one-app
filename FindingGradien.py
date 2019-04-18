import numpy as np

def FindingGradien(dx,dy):
    rotation=np.array([[np.cos(np.pi/8),-np.sin(np.pi/8)],[np.sin(np.pi/8),np.cos(np.pi/8)]])
    rotated=np.dot(rotation,np.array([[dx],[dy]]))

    if rotated[1]<0:
        rotated[0]=-rotated[0]
        rotated[1]=-rotated[1]

    s_theta=-1
    if rotated[0] >=0 and rotated[0] >= rotated[1]:
        s_theta=0
    elif rotated[0]>=0 and rotated[0]<rotated[1]:
        s_theta=1
    elif rotated[0]<0 and -rotated[0]<rotated[1]:
        s_theta=2
    elif rotated[0]<0 and -rotated[0]>=rotated[1]:
        s_theta=3

    return  s_theta