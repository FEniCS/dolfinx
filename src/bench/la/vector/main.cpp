// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
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
  static tuple<real, real> benchVectorAssign(const dolfin::uint N, const dolfin::uint size)
  {
    tuple<real, real> timing;
  
    Vec x(size);

    tic();
    for(dolfin::uint i=0; i < N; ++i)
      for(dolfin::uint j=0; j < size; ++j)
        x(j) = 1.0;      
    get<0>(timing) = toc();

    tic();
    real xtemp = 0.0;
    for(dolfin::uint i=0; i < N; ++i)
      for(dolfin::uint j=0; j < size; ++j)
        xtemp = x(j);      
    get<1>(timing) = toc();

    return timing;
  }
};
//-----------------------------------------------------------------------------
int main()
{
  // Elementwise assignment and access
  const dolfin::uint N = 100000000;
  const dolfin::uint size = 6;

  tuple<real, real> timing;

  timing = VectorAssign<uBlasVector>::benchVectorAssign(N, size);
  cout << "Time to assign values to a uBlas vector elementwise: " << get<0>(timing) << endl;
  cout << "Time to access a uBlas vector elementwise          : " << get<1>(timing) << endl;

#ifdef HAVE_PETSC_H  
  timing = VectorAssign<PETScVector>::benchVectorAssign(N, size);
  cout << "Time to assign values to a PETSc vector elementwise: " << get<0>(timing) << endl;
  cout << "Time to access a PETSc vector elementwise          : " << get<1>(timing) << endl;
#endif

  return 0;
}
