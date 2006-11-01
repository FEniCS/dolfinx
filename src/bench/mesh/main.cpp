// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-01
// Last changed: 2006-11-01

#include <dolfin.h>

using namespace dolfin;

void bench(unsigned int n, unsigned int M)
{
  real t = 0.0;
  real MM = static_cast<real>(M);

  UnitCube mesh(n, n, n);
  cout << "n = " << n << ": " << mesh << endl;

  // Create unit cube
  tic();
  for (unsigned int i = 0; i < M; ++i)
  {
    UnitCube mesh(n, n, n);
  }
  t = toc() / MM;
  dolfin_info("  Create unit cube mesh: %.3e", t);

  // Iterate over entities
  real sum = 0.0;
  tic();
  for (unsigned int i = 0; i < M; ++i)
  {
    for (CellIterator c(mesh); !c.end(); ++c)
      for (VertexIterator v(c); !v.end(); ++v)
        sum += sqrt(2.0);
  }
  t = toc() / MM;
  dolfin_info("  Iterate over entities: %.3e (sum = %g)", t, sum);
  
  // Iterate over integers
  unsigned int num_cells = mesh.numCells();
  sum = 0.0;
  tic();
  for (unsigned int i = 0; i < M; ++i)
  {
    for (unsigned int c = 0; c < num_cells; ++c)
      for (unsigned int v = 0; v < 4; ++v)
        sum += sqrt(2.0);
  }
  t = toc() / MM;
  dolfin_info("  Iterate over integers: %.3e (sum = %g)", t, sum);
  
  // Uniform refinement
  dolfin_log(false);
  tic();
  mesh.refine();
  t = toc();
  dolfin_log(true);
  dolfin_info("  Uniform refinement:    %.3e", t);
}

int main()
{
  for (unsigned int n = 1; n < 5; n++)
  {
    bench(n, 100);
  }
}
