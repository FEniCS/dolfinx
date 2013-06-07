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
// This benchmark measures the performance of building a BoundingBoxTree (and
// one call to compute_entities, which is dominated by building). The call to
// compute_entities is included so that we may compare the timing to CGAL.
//
// First added:  2013-04-18
// Last changed: 2013-05-29

#include <vector>
#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 5
#define SIZE 64

void bench_cgal()
{
  cout << "Running CGAL bench" << endl;

  // Create mesh and point to search
  UnitCubeMesh mesh(SIZE, SIZE, SIZE);
  Point point(0.5, 0.5, 0.5);

  // Compute intersection
  std::set<std::size_t> cells;
  mesh.intersected_cells(point, cells);
}

void bench_dolfin()
{
  cout << "Running DOLFIN bench" << endl;

  // Create mesh and point to search
  UnitCubeMesh mesh(SIZE, SIZE, SIZE);
  Point point(0.5, 0.5, 0.5);

  // Compute intersection
  MeshPointIntersection intersection(mesh, point);
  std::vector<unsigned int> cells = intersection.intersected_cells();
}

int main(int argc, char* argv[])
{
  // Select which benchmark to run
  bool run_cgal = argc > 1 && strcasecmp(argv[1], "cgal") == 0;

  // Run benchmark
  tic();
  for (int i = 0; i < NUM_REPS; i++)
  {
    if (run_cgal)
      bench_cgal();
    else
      bench_dolfin();
  }
  info("BENCH %g", toc());

  return 0;
}
