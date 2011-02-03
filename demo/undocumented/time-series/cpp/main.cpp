// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-11
// Last changed: 2011-02-04
//
// This program demonstrates the use of the TimeSeries
// class for storing a series of meshes and vectors.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  not_working_in_parallel("This demo");

  // Create empty time series
  TimeSeries series("primal");

  // Create a mesh and a vector
  UnitSquare unit_square(2, 2);
  Mesh mesh(unit_square);
  Vector x;

  // Add a bunch of meshes and vectors to the series
  double t = 0.0;
  while (t < 1.0)
  {
    // Refine mesh and resize vector
    mesh = refine(mesh);
    x.resize(mesh.num_vertices());

    // Set some vector values
    Array<double> values(x.local_size());
    const dolfin::uint offset = x.local_range().first;
    for (dolfin::uint i = 0; i < x.local_size(); i++)
      values[i] = (t + 1.0)*static_cast<double>(offset + i);
    x.set_local(values);
    x.apply("insert");

    // Append to series
    series.store(mesh, t);
    series.store(x, t);

    t += 0.2;
  }

  // Retrieve mesh and vector at some point in time
  series.retrieve(mesh, 0.3);
  series.retrieve(x, 0.3);

  // Plot mesh
  plot(mesh);

  return 0;
}
