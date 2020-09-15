#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>
// #include <iostream>
#include <random>


using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;


struct randn
{
    int r, c;
    randn(int r, int c):r(r),c(c){
    }
    Eigen::MatrixXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<double> dist(0,1);
        return Eigen::MatrixXd{r, c}.unaryExpr([&](double x) { return dist(gen); });
    }
};


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



int add(int i, int j) {
    return i + j;
}
PYBIND11_MODULE(nsdesolve, m) {
    m.doc() = "Stochastic simulation module"; 
    m.def("add", &add, "A function which adds two numbers");
    m.def("simulate_2d", &simulate_2d, "2D simulation, colored noise, memory kernel, magnetic field",
        "x0"_a, "y0"_a, "vx0"_a, "vy0"_a, 
        "N"_a, "samples"_a=1,"dt"_a=0.001, "warmup"_a=0,"skip"_a=1, 
        "omega0"_a,"gamma0"_a, "b"_a, "theta"_a, "kappa"_a
        , py::return_value_policy::reference_internal
    );
    m.def("simulate_1d", &simulate_1d, "1D simulation, colored noise, memory kernel",
        "x0"_a, "v0"_a,
        "N"_a, "samples"_a=1,"dt"_a=0.001, "warmup"_a=0,"skip"_a=1, 
        "omega0"_a,"gamma0"_a, "theta"_a, "kappa"_a
        , py::return_value_policy::reference_internal
    );
     m.def("simulate_1d_only_memory", &simulate_1d_only_memory, "1D simulation, only memory kernel",
        "x0"_a, "v0"_a,
        "N"_a, "samples"_a=1,"dt"_a=0.001, "warmup"_a=0,"skip"_a=1, 
        "omega0"_a,"gamma0"_a, "kappa"_a
        , py::return_value_policy::reference_internal
    );
    m.def("simulate_2d_only_memory", &simulate_2d_only_memory, "2D simulation, only memory kernel, magnetic field",
        "x0"_a, "y0"_a, "vx0"_a, "vy0"_a, 
        "N"_a, "samples"_a=1,"dt"_a=0.001, "warmup"_a=0,"skip"_a=1, 
        "omega0"_a,"gamma0"_a, "b"_a, "kappa"_a
        , py::return_value_policy::reference_internal
    );
    m.def("simulate_2d_only_memory_anharmonic_1", &simulate_2d_only_memory_anharmonic_1, "2D simulation, only memory kernel, magnetic field, anharmonic potential #1",
        "x0"_a, "y0"_a, "vx0"_a, "vy0"_a, 
        "N"_a, "samples"_a=1,"dt"_a=0.001, "warmup"_a=0,"skip"_a=1, 
        "A"_a,"B"_a, "gamma0"_a, "b"_a, "kappa"_a
        , py::return_value_policy::reference_internal
    );
}