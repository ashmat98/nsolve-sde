# ifndef ANHARMONIC_H
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
    int M = N/skip;
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

//////////////////////////////////////////

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
    int M = N/skip;
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



#endif