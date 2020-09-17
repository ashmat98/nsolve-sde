# ifndef HARMONIC_H
#define HARMONIC_H 1

#include <Eigen/Dense>
#include "myrandom.h"
#include <tuple>

using namespace std;


tuple<
Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd,Eigen::MatrixXd
> simulate_1d(
    Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> v0,
    int N, int samples, double dt, int warmup, int skip,
    double omega0, double gamma0, double theta, double kappa){
    Eigen::MatrixXd x = x0, v=v0;
    // Eigen::MatrixXd x  = Eigen::MatrixXd::Zero(samples,1), y  = Eigen::MatrixXd::Zero(samples,1);
    // Eigen::MatrixXd vx = Eigen::MatrixXd::Zero(samples,1), vy = Eigen::MatrixXd::Zero(samples,1);
    Eigen::MatrixXd r = Eigen::MatrixXd::Zero(samples,1);
    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(samples,1);
    randn rmg(samples,1);
    int M = N/skip;
    Eigen::MatrixXd  X(samples,M),  V(samples,M), 
                    R(samples,M), E(samples,M);
    double omega0_squared = omega0*omega0;
    double gamma_kappa = gamma0*kappa;
    double root_dt_theta = sqrt(dt) * theta;

    for (int i=-warmup;i<N;++i){
        x  += v * dt;
        v += (-r-omega0_squared*x + e)*dt;
        r += (- kappa * r + gamma_kappa*v)*dt;
        e += (-theta * e) * dt + root_dt_theta * rmg();

        if ((i>=0) && (i % skip == 0)){
            int j= i/skip;
            X.col(j) = x; V.col(j) = v;
            R.col(j) = r; E.col(j) = e;
        }
    }
    return tuple<Eigen::MatrixXd,Eigen::MatrixXd,
        Eigen::MatrixXd,Eigen::MatrixXd>{move(X),move(V),move(R),move(E)};
}



tuple<
Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd> simulate_1d_only_memory(
    Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> v0,
    int N, int samples, double dt, int warmup, int skip,
    double omega0, double gamma0, double kappa){
    Eigen::MatrixXd x = x0, v=v0;
    Eigen::MatrixXd r = Eigen::MatrixXd::Zero(samples,1);
    // Eigen::MatrixXd e = Eigen::MatrixXd::Zero(samples,1);
    randn rmg(samples,1);
    int M = N/skip;
    Eigen::MatrixXd  X(samples,M),  V(samples,M), 
                    R(samples,M), E(samples,M);
    double omega0_squared = omega0*omega0;
    double gamma_kappa = gamma0*kappa;
    double root_dt = sqrt(dt);

    for (int i=-warmup;i<N;++i){
        x  += v * dt;
        v += (-r-omega0_squared*x)*dt + root_dt * rmg();
        r += (- kappa * r + gamma_kappa*v)*dt;

        if ((i>=0) && (i % skip == 0)){
            int j= i/skip;
            X.col(j) = x; V.col(j) = v;
            R.col(j) = r; 
        }
    }
    return tuple<Eigen::MatrixXd,Eigen::MatrixXd,
        Eigen::MatrixXd>{move(X),move(V),move(R)};
}

tuple<
Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd,Eigen::MatrixXd
,Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd,Eigen::MatrixXd
> simulate_2d(
    Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> y0, 
    Eigen::Ref<Eigen::VectorXd> vx0, Eigen::Ref<Eigen::VectorXd> vy0, 
    int N, int samples, double dt, int warmup, int skip,
    double omega0, double gamma0, double b, double theta, double kappa){
    Eigen::MatrixXd x = x0, y = y0;
    Eigen::MatrixXd vx = vx0, vy = vy0;
    // Eigen::MatrixXd x  = Eigen::MatrixXd::Zero(samples,1), y  = Eigen::MatrixXd::Zero(samples,1);
    // Eigen::MatrixXd vx = Eigen::MatrixXd::Zero(samples,1), vy = Eigen::MatrixXd::Zero(samples,1);
    Eigen::MatrixXd rx = Eigen::MatrixXd::Zero(samples,1), ry = Eigen::MatrixXd::Zero(samples,1);
    Eigen::MatrixXd ex = Eigen::MatrixXd::Zero(samples,1), ey = Eigen::MatrixXd::Zero(samples,1);
    randn rmg(samples,1);
    int M = N/skip;
    Eigen::MatrixXd  X(samples,M),  Y(samples,M), 
                    VX(samples,M), VY(samples,M),
                    RX(samples,M), RY(samples,M),
                    EX(samples,M), EY(samples,M);
    double omega0_squared = omega0*omega0;
    double gamma_kappa = gamma0*kappa;
    double root_dt_theta = sqrt(dt) * theta;

    for (int i=-warmup;i<N;++i){
        x  += vx * dt;
        y  += vy * dt;
        vx += (-rx-omega0_squared*x + b * vy + ex)*dt;
        vy += (-ry-omega0_squared*y - b * vx + ey)*dt;
        rx += (- kappa * rx + gamma_kappa*vx)*dt;
        ry += (- kappa * ry + gamma_kappa*vy)*dt;
        ex += (-theta * ex) * dt + root_dt_theta * rmg();
        ey += (-theta * ey) * dt + root_dt_theta * rmg();

        if ((i>=0) && (i % skip == 0)){
            int j= i/skip;
             X.col(j) = x;  Y.col(j) = y;
            VX.col(j) = vx;VY.col(j) = vy;
            RX.col(j) = rx;RY.col(j) = ry;
            EX.col(j) = ex;EY.col(j) = ey;
        }
    }
    return tuple<Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd,
                 Eigen::MatrixXd,Eigen::MatrixXd
                >{move(X),move(Y),move(VX),move(VY),
                  move(RX),move(RY),move(EX),move(EY)};
//     return {X,Y,VX,VY,RX,RY,EX,EY};
//     return {move(X),move(Y),move(VX),move(VY),
//             move(RX),move(RY),move(EX),move(EY)};
}


tuple<
Eigen::MatrixXd,Eigen::MatrixXd
,Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd,Eigen::MatrixXd
> simulate_2d_only_memory(
    Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> y0, 
    Eigen::Ref<Eigen::VectorXd> vx0, Eigen::Ref<Eigen::VectorXd> vy0, 
    int N, int samples, double dt, int warmup, int skip,
    double omega0, double gamma0, double b, double kappa){
    Eigen::MatrixXd x = x0, y = y0;
    Eigen::MatrixXd vx = vx0, vy = vy0;
    Eigen::MatrixXd rx = Eigen::MatrixXd::Zero(samples,1), ry = Eigen::MatrixXd::Zero(samples,1);
    randn rmg(samples,1);
    int M = N/skip;
    Eigen::MatrixXd  X(samples,M),  Y(samples,M), 
                    VX(samples,M), VY(samples,M),
                    RX(samples,M), RY(samples,M);
    double omega0_squared = omega0*omega0;
    double gamma_kappa = gamma0*kappa;
    double root_dt = sqrt(dt);

    for (int i=-warmup;i<N;++i){
        x  += vx * dt;
        y  += vy * dt;
        vx += (-rx-omega0_squared*x + b * vy)*dt + root_dt * rmg();
        vy += (-ry-omega0_squared*y - b * vx)*dt + root_dt * rmg();
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