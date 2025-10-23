cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(sw4lite LANGUAGES CXX CUDA Fortran)

message(WARNING "This cmakefile only builds cuda version!")

find_package(MPI REQUIRED)
find_package(LAPACK REQUIRED)

add_executable(sw4lite src/main.C src/EW.C src/Source.C src/rhs4sg.C src/rhs4sg_rev.C src/SuperGrid.C src/GridPointSource.C src/time_functions_cu.C src/ew-cfromfort.C src/EW_cuda.C src/Sarray.C src/device-routines.C src/EWCuda.C src/CheckPoint.C src/Parallel_IO.C src/EW-dg.C src/MaterialData.C src/MaterialBlock.C src/Polynomial.C src/SecondOrderSection.C src/TimeSeries.C src/sacsubc.C src/curvilinear-c.C src/rhs4sgcurv.C src/rhs4sgcurv_rev.C src/addsgd.f src/bcfort.f src/boundaryOp.f src/curvilinear4sg.f src/dgmodule.f90 src/freesurfcurvisg.f src/lglnodes.f90 src/metric.f src/rhs4th3fort.f src/solerr3.f src/type_defs.f90)

target_include_directories(sw4lite PRIVATE ${MPI_CXX_INCLUDE_DIRS} tests/testil)
target_link_libraries(sw4lite PRIVATE ${MPI_CXX_LIBRARIES} ${LAPACK_LIBRARIES})
target_compile_definitions(sw4lite PRIVATE SW4_CUDA SW4_CROUTINES)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( sw4lite
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
set_source_files_properties(src/main.C src/EW.C src/Source.C src/rhs4sg.C src/rhs4sg_rev.C src/SuperGrid.C src/GridPointSource.C src/time_functions_cu.C src/ew-cfromfort.C src/EW_cuda.C src/Sarray.C src/device-routines.C src/EWCuda.C src/CheckPoint.C src/Parallel_IO.C src/EW-dg.C src/MaterialData.C src/MaterialBlock.C src/Polynomial.C src/SecondOrderSection.C src/TimeSeries.C src/sacsubc.C src/curvilinear-c.C src/rhs4sgcurv.C src/rhs4sgcurv_rev.C
  PROPERTIES LANGUAGE CUDA)

set_source_files_properties(addsgd.f  bcfort.f  boundaryOp.f  curvilinear4sg.f  dgmodule.f90  freesurfcurvisg.f  lglnodes.f90  metric.f  rhs4th3fort.f  solerr3.f  type_defs.f90 PROPERTIES LANGUAGE Fortran)
