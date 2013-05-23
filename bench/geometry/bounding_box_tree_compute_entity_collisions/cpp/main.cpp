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
// Last changed: 2013-05-23

#include <vector>
#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 1000000
#define SIZE 64

double bench_cgal(const Mesh& mesh)
{
  cout << "Running CGAL bench" << endl;

  // First call
  std::set<std::size_t> cells;
  Point point(0.0, 0.0, 0.0);
  mesh.intersected_cells(point, cells);

  // Call repeatedly
  tic();
  for (int i = 0; i < NUM_REPS; i++)
  {
    std::set<std::size_t> cells;
    point.coordinates()[0] += 1.0 / static_cast<double>(NUM_REPS);
    point.coordinates()[1] += 1.0 / static_cast<double>(NUM_REPS);
    point.coordinates()[2] += 1.0 / static_cast<double>(NUM_REPS);
    mesh.intersected_cells(point, cells);

    //for (std::set<std::size_t>::iterator it = cells.begin(); it != cells.end(); ++it)
    //  std::cout << " " << *it;
    //std::cout << std::endl;
 }

  return toc();
}

double bench_dolfin(const Mesh& mesh)
{
  cout << "Running DOLFIN bench" << endl;

  // First call
  BoundingBoxTree tree(mesh);
  tree.build();
  Point point(0.0, 0.0, 0.0);
  tree.compute_entity_collisions(point);

  // Call repeatedly
  tic();
  for (int i = 0; i < NUM_REPS; i++)
  {
    point.coordinates()[0] += 1.0 / static_cast<double>(NUM_REPS);
    point.coordinates()[1] += 1.0 / static_cast<double>(NUM_REPS);
    point.coordinates()[2] += 1.0 / static_cast<double>(NUM_REPS);
    std::vector<unsigned int> entities = tree.compute_entity_collisions(point);

    //for (unsigned int j = 0; j < entities.size(); j++)
    //  std::cout << " " << entities[j];
    //std::cout << std::endl;
  }

  return toc();
}

int main(int argc, char* argv[])
{
  // Create mesh
  UnitCubeMesh mesh(SIZE, SIZE, SIZE);

  // Select which benchmark to run
  bool run_cgal = argc > 1 && strcasecmp(argv[1], "cgal") == 0;

  // Run benchmark
  double t = 0.0;
  if (run_cgal)
    t = bench_cgal(mesh);
  else
    t = bench_dolfin(mesh);

  // Report result
  info("BENCH %g", toc());

  return 0;
}
