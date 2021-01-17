#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>

#include "harmonic_engines.h"
#include "anharmonic_engines.h"
#include "higher-order_engines.h"


using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;








int add(int i, int j) {
    return i + j;
}
PYBIND11_MODULE(nsdesolve, m) {
    m.doc() = "Stochastic simulation module"; 
    m.def("add", &add, "A function which adds two numbers");
    m.def("simulate_2d", &simulate_2d, "2D simulation, colored noise, memory kernel, magnetic field",
        "x0"_a, "y0"_a, "vx0"_a, "vy0"_a, 
        "N"_a, "samples"_a=1,"dt"_a, "warmup"_a=0, "warmup_dt"_a=0,"skip"_a=1, 
        "omega0"_a,"gamma0"_a, "b"_a, "theta"_a, "kappa"_a
        , py::return_value_policy::reference_internal
    );

    m.def("simulate_2d_strong", &simulate_2d_strong, "2D simulation, colored noise, memory kernel, magnetic field",
        "x0"_a, "y0"_a, "vx0"_a, "vy0"_a, 
        "N"_a, "samples"_a=1,"dt"_a, "warmup"_a=0, "warmup_dt"_a=0,"skip"_a=1, 
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
    m.def("simulate_3d_only_memory_anharmonic_1", &simulate_3d_only_memory_anharmonic_1, "3D simulation, only memory kernel, magnetic field, anharmonic potential #1",
        "x0"_a, "y0"_a, "z0"_a, "vx0"_a, "vy0"_a, "vz0"_a, 
        "N"_a, "samples"_a=1,"dt"_a=0.001, "warmup"_a=0,"skip"_a=1, 
        "A"_a,"B"_a, "gamma0"_a, "b"_a, "kappa"_a
        , py::return_value_policy::reference_internal
    );
    m.def("simulate_2d_only_memory_anharmonic_2", &simulate_2d_only_memory_anharmonic_2, "2D simulation, only memory kernel, magnetic field, anharmonic potential #2",
        "x0"_a, "y0"_a, "vx0"_a, "vy0"_a, 
        "N"_a, "samples"_a=1,"dt"_a=0.001, "warmup"_a=0,"skip"_a=1, 
        "A"_a,"B"_a, "C"_a,"D"_a,"F"_a, "gamma0"_a, "b"_a, "kappa"_a
        , py::return_value_policy::reference_internal
    );
    
    m.def("simulate_2d_anharmonic_multinoise", 
        &simulate_2d_anharmonic_multinoise, 
        "2D simulation, memory kernel, colored and white noise, magnetic field, anharmonic potential #2",
        "x0"_a, "y0"_a, "vx0"_a, "vy0"_a, 
        "N"_a, "samples"_a=1,"dt"_a=0.001, "warmup"_a=0,"skip"_a=1, 
        "A"_a,"B"_a, "C"_a,"D"_a,"F"_a, "gamma0"_a, "b"_a, "kappa"_a,
        "theta"_a, "q_colored"_a, "q_white"_a
        , py::return_value_policy::reference_internal
    );

    m.def("Uxyz_1", &Uxyz_1, "Nonlinear potential gradient",
        "A"_a,"B"_a,"x"_a,"y"_a,"z"_a
        , py::return_value_policy::reference_internal
    );
    m.def("Uxy_2", &Uxy_2, "Nonlinear potential gradient",
        "A"_a,"B"_a,"C"_a,"D"_a,"F"_a,"x"_a,"y"_a
        , py::return_value_policy::reference_internal
    );
    m.def("U_2", &U_2, "Nonlinear potential gradient",
        "A"_a,"B"_a,"C"_a,"D"_a,"F"_a,"x"_a,"y"_a
        , py::return_value_policy::reference_internal
    );
}