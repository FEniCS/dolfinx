# CheckOpenMP - this module defines the following macros

# check_openmp_unsigned_int_loop_control_variable(<var>)
#  <var> - variable to store the result
# This macro checks if the control variable of a loop associated
# with a for directive can be of unsigned type. In OpenMP 2.5,
# only signed integer type was allowed. See Section 2.5.1 on
# p. 38 of the OpenMP 3.0 Specification.

include(CheckCXXSourceRuns)

macro(check_openmp_unsigned_int_loop_control_variable _test_result)
  if (NOT OPENMP_FOUND)
    find_package(OpenMP)
  endif()

  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${OpenMP_CXX_FLAGS}")

  check_cxx_source_runs("
#include <iostream>
#include <omp.h>

#define N 20

int main ()
{
  float a[N];

  omp_set_dynamic(0);
  omp_set_num_threads(10);

  for (int i=0; i<N; ++i) {
    a[i] = i;
  }

#pragma omp parallel shared(a)
  {
    #pragma omp for
    for (unsigned int i=0; i<N; ++i) {
      a[i] = a[i] * 2.0;
    }

    #pragma omp for
    for (unsigned long j=0; j<N; ++j) {
      a[j] = a[j] * 2.0;
    }
  }

  for (int i=0; i<N; i++) {
    std::cout << i << \": \" << a[i] << std::endl;
  }
}
" ${_test_result})

endmacro()
