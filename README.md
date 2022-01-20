# Fortran tensor benchmarking
A simple script to benchmark, and hopefully illustrate good practices of tensor operations in Modern Fortran

## Compilation notes
### gfortran (tested on 9.3.0)
You need an installation of OpenBLAS that is compiled with `USE_OPENMP=1`, see [here](https://github.com/brianz98/A-Fortran-Electronic-Structure-Programme) for notes
```
gfortran dgemm_test.f90 -o dgemm_test -lopenblas -fopenmp -O3
```
### ifort (tested on 2021.5.0)
```
ifort dgemm_test.f90 -o dgemm_test  -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl  -i8  -I"${MKLROOT}/include" -fopenmp -O3 -heap-arrays
```
