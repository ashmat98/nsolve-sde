g++ -O2 -Wall -shared -std=c++14 -fPIC \
`python -m pybind11 --includes` \
-I/home/ashot/eigen/  \
-I/home/ashot/pybind11/include  \
-I/home/ashot/miniconda3/include/python3.7m \
-L/home/ashot/miniconda3/lib \
./stochastic-nsolve.cpp \
-o nsdesolve`python -c "from distutils import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"`