# ifndef HIGHER_H
#define HIGHER_H 1

#include <Eigen/Dense>
#include "myrandom.h"
#include <tuple>

using namespace std;

tuple<
Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd,Eigen::MatrixXd
,Eigen::MatrixXd,Eigen::MatrixXd,
Eigen::MatrixXd,Eigen::MatrixXd
> simulate_2d_strong(
    Eigen::Ref<Eigen::VectorXd> x0, Eigen::Ref<Eigen::VectorXd> y0, 
    Eigen::Ref<Eigen::VectorXd> vx0, Eigen::Ref<Eigen::VectorXd> vy0, 
    int N, int samples, double dt, int warmup, double warmup_dt, int skip,
    double omega0, double gamma0, double b, double theta, double kappa){
    Eigen::MatrixXd x = x0, y = y0;
    Eigen::MatrixXd vx = vx0, vy = vy0;
    // Eigen::MatrixXd x  = Eigen::MatrixXd::Zero(samples,1), y  = Eigen::MatrixXd::Zero(samples,1);
    // Eigen::MatrixXd vx = Eigen::MatrixXd::Zero(samples,1), vy = Eigen::MatrixXd::Zero(samples,1);
    Eigen::MatrixXd rx = Eigen::MatrixXd::Zero(samples,1), ry = Eigen::MatrixXd::Zero(samples,1);
    Eigen::MatrixXd ex = Eigen::MatrixXd::Zero(samples,1), ey = Eigen::MatrixXd::Zero(samples,1);
    randn rmg(samples,1);
    int M = (N-1)/skip+1;
    Eigen::MatrixXd  X(samples,M),  Y(samples,M), 
                    VX(samples,M), VY(samples,M),
                    RX(samples,M), RY(samples,M),
                    EX(samples,M), EY(samples,M);
    double omega0_squared = omega0*omega0;
    double gamma_kappa = gamma0*kappa;
    double _dt = warmup_dt;
    double _dt_sqr = _dt*_dt;
    double root_dt_theta = sqrt(_dt*gamma0*2) * theta;
    const double invroot3 = 1/sqrt(3);

    Eigen::MatrixXd Wa,Wb,Ua,Ub;

    for (int i=-warmup;i<N;++i){
        if (i==0){
            _dt = dt;
            _dt_sqr = _dt*_dt;
            root_dt_theta = sqrt(dt*gamma0*2) * theta;
        }

        Wa = rmg();Wb=rmg();Ua=rmg();Ub=rmg();

        x  += vx * _dt - 0.5 * _dt_sqr * (rx - b * vy - ex + x * omega0_squared);
        y  += vy * _dt - 0.5 * _dt_sqr * (ry + b * vx - ey + y * omega0_squared);
  
        vx += _dt*((-rx-x * omega0_squared + b * vy + ex) + 0.5 * (Ua*invroot3 + Wa) * root_dt_theta)
            - 0.5 * _dt_sqr * ( b * ry + b*b * vx - b * ey + ex * theta - rx * kappa + vx * gamma0 * kappa + (vx + b*y) * omega0_squared);

        vy+= _dt *((-ry - y * omega0_squared - b*vx + ey) + 0.5 * (Ub*invroot3 + Wb) * root_dt_theta)
            - 0.5 * _dt_sqr * (-b * rx + b*b * vy + b * ex + ey * theta - ry * kappa + vy * gamma0 * kappa + (vy - b*x) * omega0_squared);

        rx += (- kappa * rx + gamma_kappa*vx)*_dt
            + 0.5 * _dt_sqr * (rx * (kappa*kappa - gamma_kappa) + gamma_kappa * (   b * vy + ex - vx * kappa) - x * gamma_kappa * omega0_squared);
        ry += (- kappa * ry + gamma_kappa*vy)*_dt
            + 0.5 * _dt_sqr * (ry * (kappa*kappa - gamma_kappa) + gamma_kappa * ( - b * vx + ey - vy * kappa) - y * gamma_kappa * omega0_squared);

        ex += (-theta * ex) * _dt + root_dt_theta * Wa
            + 0.5 * _dt * theta * (_dt * theta * ex - root_dt_theta * (Wa + invroot3 * Ua));

        ey += (-theta * ey) * _dt + root_dt_theta * Wb;
            + 0.5 * _dt * theta * (_dt * theta * ey - root_dt_theta * (Wb + invroot3 * Ub));


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


#endif