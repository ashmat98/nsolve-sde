for /f %%i in ('python -c "from distutils import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set PyEXT=%%i
for /f %%i in ('python -m pybind11 --includes') do set PyINCLUDES=%%i

REM g++  -shared eigen-example.cpp 
g++ -O2 -mavx2 -ffast-math -Wall -shared -std=c++14 -fPIC ^
%PyINCLUDES% ^
-IC:\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\lib\gcc\x86_64-w64-mingw32\8.1.0\include\ ^
-LC:\Users\Ashot\Miniconda3\libs ^
stochastic-nsolve.cpp ^
-lpython38 ^
-o nsdesolve%PyEXT%