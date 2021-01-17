g++ -O2 -mavx2 -ffast-math -Wall -shared -std=c++14 -fPIC \
`python -m pybind11 --includes` \
-I/home/ashot/eigen/  \
-I/home/ashot/pybind11/include  \
-L/home/ashot/miniconda3/lib \
-lpython3.7m \
./cpp/stochastic-nsolve.cpp \
-o nsdesolve`python -c "from distutils import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"`
