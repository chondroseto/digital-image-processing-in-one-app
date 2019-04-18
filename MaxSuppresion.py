def MaxSuppesion (E_mag, i,j, s_theta, t_low):
    magC=E_mag[i,j]
    if magC<t_low:
        return False
    else:
        magL=-1
        if s_theta==0:
            magL=E_mag[i,j-1]
        elif s_theta==1:
            magL=E_mag[i-1,j-1]
        elif s_theta==2:
            magL=E_mag[i-1,j]
        elif s_theta==3:
            magL=E_mag[i-1,j+1]

        magR=-1
        if s_theta==0:
            magL=E_mag[i,j+1]
        elif s_theta==1:
            magL=E_mag[i+1,j+1]
        elif s_theta==2:
            magL=E_mag[i+1,j]
        elif s_theta==3:
            magL=E_mag[i+1,j-1]

    return magL<=magC and magC>=magR