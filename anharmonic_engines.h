#ifndef ANHARMONIC_H
#define ANHARMONIC_H 1

#include <Eigen/Dense>
#include "myrandom.h"
#include <tuple>

using namespace std;

tuple<Eigen::VectorXd, Eigen::VectorXd> Uxy_1(double A, double B, 
//                            Eigen::VectorXd x,  Eigen::VectorXd y){
    const Eigen::VectorXd & x, const Eigen::VectorXd & y){
    /// U(x,y) = V(x^2 + y^2)
    /// V(r^2) = 1/4 B r^4 - 1/2 A r^2
    
    Eigen::ArrayXd r_squared = x.array().square() + y.array().square();
    Eigen::ArrayXd common = (B*r_squared - A);
    Eigen::VectorXd ux = (x.array() * common).matrix();
    Eigen::VectorXd uy = (y.array() * common).matrix();
    return tuple<Eigen::VectorXd,Eigen::VectorXd>(move(ux),move(uy));
//     return tuple<Eigen::VectorXd,Eigen::VectorXd>(move(x),move(y));

}

tuple<
Eigen::MatrixXd,Eigen::MatrixXd
,Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd,Eigen::MatrixXd
> simulate_2d_only_memory_anharmonic_1(
    Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> y0, 
    Eigen::Ref<Eigen::VectorXd> vx0, Eigen::Ref<Eigen::VectorXd> vy0, 
    int N, int samples, double dt, int warmup, int skip,
    double A, double B, double gamma0, double b, double kappa){
    Eigen::MatrixXd x = x0, y = y0;
    Eigen::MatrixXd vx = vx0, vy = vy0;
    Eigen::MatrixXd rx = Eigen::MatrixXd::Zero(samples,1), ry = Eigen::MatrixXd::Zero(samples,1);
    randn rmg(samples,1);
    int M = (N-1)/skip+1;
    Eigen::MatrixXd  X(samples,M),  Y(samples,M), 
                    VX(samples,M), VY(samples,M),
                    RX(samples,M), RY(samples,M);
    double gamma_kappa = gamma0*kappa;
    double root_dt = sqrt(dt);

    for (int i=-warmup;i<N;++i){
        x  += vx * dt;
        y  += vy * dt;
        tuple<Eigen::VectorXd, Eigen::VectorXd> uxy = Uxy_1(A,B,x,y);
//         tuple<Eigen::VectorXd, Eigen::VectorXd> uxy = make_tuple(Eigen::MatrixXd::Zero(samples,1),Eigen::MatrixXd::Zero(samples,1));

        vx += (-rx - get<0>(uxy) + b * vy)*dt + root_dt * rmg();
        vy += (-ry - get<1>(uxy) - b * vx)*dt + root_dt * rmg();
        rx += (- kappa * rx + gamma_kappa*vx)*dt;
        ry += (- kappa * ry + gamma_kappa*vy)*dt;

        if ((i>=0) && (i % skip == 0)){
            int j= i/skip;
             X.col(j) = x;  Y.col(j) = y;
            VX.col(j) = vx;VY.col(j) = vy;
            RX.col(j) = rx;RY.col(j) = ry;
        }
    }
    return tuple<Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd
                >{move(X),move(Y),move(VX),move(VY),
                  move(RX),move(RY)};
}


tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> Uxyz_1(double A, double B, 
//                            Eigen::VectorXd x,  Eigen::VectorXd y){
    const Eigen::VectorXd & x, const Eigen::VectorXd & y, const Eigen::VectorXd & z){
    /// U(x,y) = V(x^2 + y^2 + z^2)
    /// V(r^2) = 1/4 B r^4 - 1/2 A r^2
    
    Eigen::ArrayXd r_squared = x.array().square() + y.array().square() + z.array().square();
    Eigen::ArrayXd common = (B*r_squared - A);
    Eigen::VectorXd ux = (x.array() * common).matrix();
    Eigen::VectorXd uy = (y.array() * common).matrix();
    Eigen::VectorXd uz = (z.array() * common).matrix();
    return tuple<Eigen::VectorXd,Eigen::VectorXd,Eigen::VectorXd>(move(ux),move(uy),move(uz));
//     return tuple<Eigen::VectorXd,Eigen::VectorXd>(move(x),move(y));

}

tuple<
Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd
,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd
> simulate_3d_only_memory_anharmonic_1(
    Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> y0, Eigen::Ref<Eigen::VectorXd> z0, 
    Eigen::Ref<Eigen::VectorXd> vx0, Eigen::Ref<Eigen::VectorXd> vy0, Eigen::Ref<Eigen::VectorXd> vz0, 
    int N, int samples, double dt, int warmup, int skip,
    double A, double B, double gamma0, double b, double kappa){
    Eigen::MatrixXd x = x0, y = y0, z=z0;
    Eigen::MatrixXd vx = vx0, vy = vy0, vz=vz0;
    Eigen::MatrixXd rx = Eigen::MatrixXd::Zero(samples,1), 
                    ry = Eigen::MatrixXd::Zero(samples,1),
                    rz = Eigen::MatrixXd::Zero(samples,1);
    randn rmg(samples,1);
    int M = (N-1)/skip+1;
    Eigen::MatrixXd  X(samples,M),  Y(samples,M),  Z(samples,M), 
                    VX(samples,M), VY(samples,M), VZ(samples,M),
                    RX(samples,M), RY(samples,M), RZ(samples,M);
    double gamma_kappa = gamma0*kappa;
    double root_dt = sqrt(dt);

    for (int i=-warmup;i<N;++i){
        x  += vx * dt;
        y  += vy * dt;
        z  += vz * dt;
        tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd
              > uxyz = Uxyz_1(A,B,x,y,z);

        vx += (-rx - get<0>(uxyz) + b * vy)*dt + root_dt * rmg();
        vy += (-ry - get<1>(uxyz) - b * vx)*dt + root_dt * rmg();
        vz += (-rz - get<2>(uxyz) )*dt + root_dt * rmg();
        rx += (- kappa * rx + gamma_kappa*vx)*dt;
        ry += (- kappa * ry + gamma_kappa*vy)*dt;
        rz += (- kappa * rz + gamma_kappa*vz)*dt;

        if ((i>=0) && (i % skip == 0)){
            int j= i/skip;
             X.col(j) = x;  Y.col(j) = y;  Z.col(j) = z;
            VX.col(j) = vx;VY.col(j) = vy;VZ.col(j) = vz;
            RX.col(j) = rx;RY.col(j) = ry;RZ.col(j) = rz;
        }
    }
    return tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd
                >{move(X),move(Y),move(Z),move(VX),move(VY),move(VZ),
                  move(RX),move(RY),move(RZ)};
}

//////////////////////////////////////////

Eigen::VectorXd U_2(double A, double B, double C, double D, double F,
    const Eigen::VectorXd & x, const Eigen::VectorXd & y){
    /// U(x,y) = V(x^2 + y^2)
    /// V(r^2) = 1/4 B r^4 - 1/2 A r^2 + C cos(Fr) exp(-Dr^2)

    Eigen::ArrayXd r_squared = x.array().square() + y.array().square();
    Eigen::ArrayXd r = r_squared.sqrt();
    // return (-1.0/2.0*A*r_squared).matrix();
    return (1.0/4.0*B*r_squared*r_squared - 1.0/2.0*A*r_squared + 
        C*Eigen::cos(F*r)*Eigen::exp(-D*r_squared)).matrix();
}


tuple<Eigen::VectorXd, Eigen::VectorXd> Uxy_2(double A, double B, double C, double D, double F,
//                            Eigen::VectorXd x,  Eigen::VectorXd y){
    const Eigen::VectorXd & x, const Eigen::VectorXd & y){
    /// U(x,y) = V(x^2 + y^2)
    /// V(r^2) = 1/4 B r^4 - 1/2 A r^2 + C cos(Fr) exp(-Dr^2)
    static double EPS = 1e-7;

    Eigen::ArrayXd r_squared = x.array().square() + y.array().square();
    Eigen::ArrayXd Fr = F*r_squared.sqrt();

    Eigen::ArrayXd common = (B*r_squared - A)
        - Eigen::exp(-D*r_squared)*(2*C*D*Eigen::cos(Fr) - C*F*F*(EPS+Eigen::sin(Fr))/(EPS+Fr)) ;
    Eigen::VectorXd ux = (x.array() * common).matrix();
    Eigen::VectorXd uy = (y.array() * common).matrix();
    return tuple<Eigen::VectorXd,Eigen::VectorXd>(move(ux),move(uy));
}

tuple<
Eigen::MatrixXd,Eigen::MatrixXd
,Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd,Eigen::MatrixXd
> simulate_2d_only_memory_anharmonic_2(
    Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> y0, 
    Eigen::Ref<Eigen::VectorXd> vx0, Eigen::Ref<Eigen::VectorXd> vy0, 
    int N, int samples, double dt, int warmup, int skip,
    double A, double B, double C, double D, double F, 
    double gamma0, double b, double kappa){
    Eigen::MatrixXd x = x0, y = y0;
    Eigen::MatrixXd vx = vx0, vy = vy0;
    Eigen::MatrixXd rx = Eigen::MatrixXd::Zero(samples,1), ry = Eigen::MatrixXd::Zero(samples,1);
    randn rmg(samples,1);
    int M = (N-1)/skip+1;
    Eigen::MatrixXd  X(samples,M),  Y(samples,M), 
                    VX(samples,M), VY(samples,M),
                    RX(samples,M), RY(samples,M);
    double gamma_kappa = gamma0*kappa;
    double root_dt = sqrt(dt);

    for (int i=-warmup;i<N;++i){
        x  += vx * dt;
        y  += vy * dt;
        tuple<Eigen::VectorXd, Eigen::VectorXd> uxy = Uxy_2(A,B,C,D,F,x,y);
//         tuple<Eigen::VectorXd, Eigen::VectorXd> uxy = make_tuple(Eigen::MatrixXd::Zero(samples,1),Eigen::MatrixXd::Zero(samples,1));

        vx += (-rx - get<0>(uxy) + b * vy)*dt + root_dt * rmg();
        vy += (-ry - get<1>(uxy) - b * vx)*dt + root_dt * rmg();
        rx += (- kappa * rx + gamma_kappa*vx)*dt;
        ry += (- kappa * ry + gamma_kappa*vy)*dt;

        if ((i>=0) && (i % skip == 0)){
            int j= i/skip;
             X.col(j) = x;  Y.col(j) = y;
            VX.col(j) = vx;VY.col(j) = vy;
            RX.col(j) = rx;RY.col(j) = ry;
        }
    }
    return tuple<Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd
                >{move(X),move(Y),move(VX),move(VY),
                  move(RX),move(RY)};
}


tuple<
Eigen::MatrixXd,Eigen::MatrixXd
,Eigen::MatrixXd,Eigen::MatrixXd
> simulate_2d_anharmonic_multinoise(
    Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> y0, 
    Eigen::Ref<Eigen::VectorXd> vx0, Eigen::Ref<Eigen::VectorXd> vy0, 
    int N, int samples, double dt, int warmup, int skip,
    double A, double B, double C, double D, double F, 
    double gamma0, double b, double kappa, double theta,
    double q_colored, double q_white){
    /**
     * single friction with memory kernel described by gamma0, kappa
     * and two fluctuation forces, colored noise with q=1 and
     * white noise with q = q_qhite
     **/
    Eigen::MatrixXd x = x0, y = y0;
    Eigen::MatrixXd vx = vx0, vy = vy0;
    Eigen::MatrixXd rx = Eigen::MatrixXd::Zero(samples,1), ry = Eigen::MatrixXd::Zero(samples,1);
    Eigen::MatrixXd ex = Eigen::MatrixXd::Zero(samples,1), ey = Eigen::MatrixXd::Zero(samples,1);

    randn rmg(samples,1);
    int M = (N-1)/skip+1;
    Eigen::MatrixXd  X(samples,M),  Y(samples,M), 
                    VX(samples,M), VY(samples,M);
    double gamma_kappa = gamma0*kappa;
    double root_dt_theta = sqrt(dt*theta*q_colored);
    double root_q_dt = sqrt(q_white*dt);


    for (int i=-warmup;i<N;++i){
        x  += vx * dt;
        y  += vy * dt;
        tuple<Eigen::VectorXd, Eigen::VectorXd> uxy = Uxy_2(A,B,C,D,F,x,y);
//         tuple<Eigen::VectorXd, Eigen::VectorXd> uxy = make_tuple(Eigen::MatrixXd::Zero(samples,1),Eigen::MatrixXd::Zero(samples,1));

        vx += (-rx - get<0>(uxy) + b * vy + ex)*dt + root_q_dt * rmg();
        vy += (-ry - get<1>(uxy) - b * vx + ey)*dt + root_q_dt * rmg();
        rx += (- kappa * rx + gamma_kappa*vx)*dt;
        ry += (- kappa * ry + gamma_kappa*vy)*dt;
        ex += (-theta * ex) * dt + root_dt_theta * rmg();
        ey += (-theta * ey) * dt + root_dt_theta * rmg();

        if ((i>=0) && (i % skip == 0)){
            int j= i/skip;
             X.col(j) = x;  Y.col(j) = y;
            VX.col(j) = vx;VY.col(j) = vy;
        }
    }
    return tuple<Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd
                >{move(X),move(Y),move(VX),move(VY)};
}



#endif