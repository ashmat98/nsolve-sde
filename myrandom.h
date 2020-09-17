#ifndef MYRANDOM_H
#define MYRANDOM_H 1
#include <Eigen/Dense>
#include <random>

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

#endif