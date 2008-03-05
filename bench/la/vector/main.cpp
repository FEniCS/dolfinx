// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-08-18
// Last changed:
//
// Benchmarks for assigning values to vector

#include <dolfin.h>
#include <boost/tuple/tuple.hpp>

using namespace dolfin;
using namespace boost::tuples;

//-----------------------------------------------------------------------------
template<class Vec>
struct VectorAssign
{
  static tuple<real, real> benchVectorAssign(const dolfin::uint N, const dolfin::uint n)
  {
    set("output destination", "silent");

    tuple<real, real> timing;
    Vec x(N);

    tic();
    for(dolfin::uint i=0; i < n; ++i)
      for(dolfin::uint j=0; j < N; ++j)
        x(j) = 1.0;      
    get<0>(timing) = toc();

    tic();
    real xtemp = 0.0;
    for(dolfin::uint i=0; i < n; ++i)
      for(dolfin::uint j=0; j < N; ++j)
        xtemp = x(j);      
    get<1>(timing) = toc();

    return timing;
  }
};
//-----------------------------------------------------------------------------
int main()
{
  // Bechmark elementwise assignment and access
  const dolfin::uint N[3] = {6, 6, 100};
  const dolfin::uint n[3] = {100000, 100000000, 1000000};

  tuple<real, real> ublas_timing[3];
#ifdef HAS_PETSC  
  tuple<real, real> petsc_timing[2];
#endif

  begin("Vector benchmark timings");

  // Perform uBlas benchmarks
  ublas_timing[0] = VectorAssign<uBlasVector>::benchVectorAssign(N[0], n[0]);
  ublas_timing[1] = VectorAssign<uBlasVector>::benchVectorAssign(N[1], n[1]);
  ublas_timing[2] = VectorAssign<uBlasVector>::benchVectorAssign(N[2], n[2]);

#ifdef HAS_PETSC  
  // Perform PETSc benchmarks
  petsc_timing[0] = VectorAssign<PETScVector>::benchVectorAssign(N[0], n[0]);
#endif

  // Output assignment timings
  set("output destination", "terminal");
  begin("Assign values to a vector of length N elementwise n times");
#ifdef HAS_PETSC  
  cout << "PETScVector (N="<< N[0] << ", n=" << n[0] << "): " << get<0>(petsc_timing[0]) << endl;
#endif
  for(dolfin::uint i=0; i< 3; ++i)
    cout << "uBlasVector (N="<< N[i] << ", n=" << n[i] << "): " << get<0>(ublas_timing[i]) << endl;

  end();

  // Output access timings
  begin("Access values of a vector of length n elementwise N times");
#ifdef HAS_PETSC  
  cout << "PETScVector (N="<< N[0] << ", n=" << n[0] << "): " << get<0>(petsc_timing[0]) << endl;
#endif
  for(dolfin::uint i=0; i< 3; ++i)
    cout << "uBlasVector (N="<< N[i] << ", n=" << n[i] << "): " << get<1>(ublas_timing[i]) << endl;

  end();
  end();

  return 0;
}
