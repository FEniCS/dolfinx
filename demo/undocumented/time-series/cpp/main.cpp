// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-11-11
// Last changed: 2012-11-12
//
// This program demonstrates the use of the TimeSeries
// class for storing a series of meshes and vectors.

#include <dolfin.h>

using namespace dolfin;

int main()
{
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    std::cout << "This demo does not work in parallel" << std::endl;
    return 0;
  }

  #ifdef HAS_HDF5

  // Create empty time series
  TimeSeries series(MPI_COMM_WORLD, "primal");

  // Create a mesh and a vector
  UnitSquareMesh unit_square(2, 2);
  Mesh mesh(unit_square);

  // Add a bunch of meshes and vectors to the series
  double t = 0.0;
  while (t < 1.0)
  {
    // Refine mesh and resize vector
    mesh = refine(mesh);
    Vector x(mesh.mpi_comm(), mesh.num_vertices());

    // Set some vector values
    std::vector<double> values(x.local_size());
    const std::size_t offset = x.local_range().first;
    for (std::size_t i = 0; i < x.local_size(); i++)
      values[i] = (t + 1.0)*static_cast<double>(offset + i);
    x.set_local(values);
    x.apply("insert");

    // Append to series
    series.store(mesh, t);
    series.store(x, t);

    t += 0.2;
  }

  // Retrieve mesh and vector at some point in time
  series.retrieve(mesh, 0.29);
  Vector x;
  series.retrieve(x, 0.31, false);

  #endif

  return 0;
}
