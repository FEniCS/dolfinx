// Copyright (C) 2013 Anders Logg
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
// This benchmark measures the performance of compute_entity_collisions.
//
// First added:  2013-05-23
// Last changed: 2013-09-04

#include <vector>
#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 1000000
#define SIZE 64

int main(int argc, char* argv[])
{
  info("Compute closest entity on UnitCubeMesh(%d, %d, %d)",
       SIZE, SIZE, SIZE);

  // Create mesh
  UnitCubeMesh mesh(SIZE, SIZE, SIZE);

  // First call
  BoundingBoxTree tree;
  tree.build(mesh);
  Point point(-1.0, -1.0, 0.0);
  tree.compute_closest_entity(point);
  cout << "Built tree, searching for closest point" << endl;

  // Call repeatedly
  tic();
  for (int i = 0; i < NUM_REPS; i++)
  {
    tree.compute_closest_entity(point);
    point.coordinates()[1] += 2.0 / static_cast<double>(NUM_REPS);
  }
  const double t = toc();

  // Report result
  info("BENCH %g", t);

  return 0;
}
