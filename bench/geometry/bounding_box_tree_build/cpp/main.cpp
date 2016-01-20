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
// one call to compute_entities, which is dominated by building).
//
// First added:  2013-04-18
// Last changed: 2013-06-25

#include <vector>
#include <dolfin.h>

using namespace dolfin;

#define SIZE 128

int main(int argc, char* argv[])
{
  info("Build bounding box tree on UnitCubeMesh(%d, %d, %d)",
       SIZE, SIZE, SIZE);

  // Create mesh
  UnitCubeMesh mesh(SIZE, SIZE, SIZE);

  // Create and build tree
  tic();
  BoundingBoxTree tree;
  tree.build(mesh);
  info("BENCH %g", toc());

  return 0;
}
