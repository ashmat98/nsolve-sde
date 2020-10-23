import numpy as np

def Uxyz_1(A, B, x, y, z):
    ### U(x,y) = V(x^2 + y^2)
    ### V(r^2) = 1/4 B r^4 - 1/2 A r^2 + C cos(Fr) exp(-Dr^2)
    EPS = 1e-7

    r_squared = np.square(x) + np.square(y) + np.square(z)

    common = (B*r_squared + A) 
    ux = (x * common)
    uy = (y * common)
    uz = (z * common)
    return ux, uy, uz

def simulate_3d_only_memory_anharmonic_1_py(
    x0, y0, z0,
    vx0, vy0, vz0,
     N,  samples,  dt,  warmup,  skip,
     A,  B,
     gamma0,  b,  kappa):
    x = x0.copy()[:,None]; y = y0.copy()[:,None]; z = z0.copy()[:,None]
    vx = vx0.copy()[:,None]; vy = vy0.copy()[:,None]; vz = vz0.copy()[:,None]
    rx = np.zeros((samples,1)); ry = np.zeros((samples,1)); rz = np.zeros((samples,1))
    M = (N-1)//skip+1
    X = np.zeros((samples,M))
    Y = np.zeros((samples,M))
    Z = np.zeros((samples,M))

    VX = np.zeros((samples,M))
    VY = np.zeros((samples,M))
    VZ = np.zeros((samples,M))

    RX = np.zeros((samples,M))
    RY = np.zeros((samples,M))
    RZ = np.zeros((samples,M))
    
    gamma_kappa = gamma0*kappa
    root_dt = np.sqrt(dt)

    for i in range(N):
        x  += vx * dt
        y  += vy * dt
        z  += vz * dt
        ux, uy, uz = Uxyz_1(A,B,x,y,z)

        vx += (-rx - ux + b * vy)*dt + root_dt * np.random.randn(samples,1)
        vy += (-ry - uy - b * vx)*dt + root_dt * np.random.randn(samples,1)
        vz += (-rz - uz         )*dt + root_dt * np.random.randn(samples,1)

        rx += (- kappa * rx + gamma_kappa*vx)*dt
        ry += (- kappa * ry + gamma_kappa*vy)*dt
        rz += (- kappa * rz + gamma_kappa*vz)*dt
        if (i>=0) and (i % skip == 0):
            j= i//skip
            X[:,j] = x.flatten();    Y[:,j] = y.flatten() ;  Z[:,j] = z.flatten()
            VX[:,j] = vx.flatten(); VY[:,j] = vy.flatten(); VZ[:,j] = vz.flatten()
            RX[:,j] = rx.flatten(); RY[:,j] = ry.flatten(); RZ[:,j] = rz.flatten() 
        
    
    return X,Y,Z,VX,VY,VZ,RX,RY,RZ



def Uxy_2(A, B, C, D, F, x, y):
    ### U(x,y) = V(x^2 + y^2)
    ### V(r^2) = 1/4 B r^4 - 1/2 A r^2 + C cos(Fr) exp(-Dr^2)
    EPS = 1e-7

    r_squared = np.square(x) + np.square(y)
    Fr = F*np.sqrt(r_squared)

    common = (B*r_squared + A) \
        - np.exp(-D*r_squared)*(2*C*D*np.cos(Fr) - C*F*F*(EPS+np.sin(Fr))/(EPS+Fr))
    ux = (x * common)
    uy = (y * common)
    return ux, uy


def simulate_2d_only_memory_anharmonic_2_py(
    x0, y0, 
    vx0, vy0, 
     N,  samples,  dt,  warmup,  skip,
     A,  B,  C,  D,  F, 
     gamma0,  b,  kappa):
    x = x0.copy()[:,None]; y = y0.copy()[:,None]
    vx = vx0.copy()[:,None]; vy = vy0.copy()[:,None]
    rx = np.zeros((samples,1)); ry = np.zeros((samples,1))
    M = N//skip
    X = np.zeros((samples,M))
    Y = np.zeros((samples,M))
    VX = np.zeros((samples,M))
    VY = np.zeros((samples,M))
    RX = np.zeros((samples,M))
    RY = np.zeros((samples,M))
    
    gamma_kappa = gamma0*kappa
    root_dt = np.sqrt(dt)

    for i in range(N):
        x  += vx * dt
        y  += vy * dt
        ux, uy = Uxy_2(A,B,C,D,F,x,y)

        vx += (-rx - ux + b * vy)*dt + root_dt * np.random.randn(samples,1)
        vy += (-ry - uy - b * vx)*dt + root_dt * np.random.randn(samples,1)
        rx += (- kappa * rx + gamma_kappa*vx)*dt
        ry += (- kappa * ry + gamma_kappa*vy)*dt

        if (i>=0) and (i % skip == 0):
            j= i//skip
            X[:,j] = x.flatten();    Y[:,j] = y.flatten()
            VX[:,j] = vx.flatten(); VY[:,j] = vy.flatten()
            RX[:,j] = rx.flatten(); RY[:,j] = ry.flatten()
        
    
    return X,Y,VX,VY,RX,RY
