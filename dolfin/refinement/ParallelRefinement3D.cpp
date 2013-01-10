// Copyright (C) 2012 Chris Richardson
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
// 
// First Added: 2012-12-19
// Last Changed: 2013-01-10

#include <vector>
#include <map>
#include <boost/unordered_map.hpp>
#include <boost/multi_array.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/LocalMeshData.h>

#include <dolfin/refinement/ParallelRefinement.h>

#include "ParallelRefinement3D.h"

using namespace dolfin;


// Uniform refinement

void ParallelRefinement3D::refine(Mesh& new_mesh, const Mesh& mesh)
{
  if(MPI::num_processes()==1)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in parallel");
  }

  const uint tdim = mesh.topology().dim();
  const uint gdim = mesh.geometry().dim();

  if(tdim != 3 || gdim != 3)
  {
    dolfin_error("ParallelRefinement3D.cpp",
                 "refine mesh",
                 "Only works in 3D");
  }

  // Ensure connectivity is there etc
  mesh.init(1);
  mesh.init(1, tdim);

  std::cout << "Num edges = " << mesh.num_edges() << std::endl;

  // Instantiate a class to hold most of the refinement information
  ParallelRefinement p(mesh);
  
  // Mark all edges, and create new vertices
  EdgeFunction<bool> markedEdges(mesh, true);
  p.create_new_vertices(markedEdges);
  std::map<std::size_t, std::size_t>& edge_to_new_vertex = p.edge_to_new_vertex();
  
  // Generate new topology

  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {
    EdgeIterator e(*cell);
    VertexIterator v(*cell);

    //     std::cout << cell->index() << std::endl;
    

    const std::size_t v0 = v[0].global_index();
    const std::size_t v1 = v[1].global_index();
    const std::size_t v2 = v[2].global_index();
    const std::size_t v3 = v[3].global_index();
    const std::size_t e0 = edge_to_new_vertex[e[0].index()];
    const std::size_t e1 = edge_to_new_vertex[e[1].index()];
    const std::size_t e2 = edge_to_new_vertex[e[2].index()];
    const std::size_t e3 = edge_to_new_vertex[e[3].index()];
    const std::size_t e4 = edge_to_new_vertex[e[4].index()];
    const std::size_t e5 = edge_to_new_vertex[e[5].index()];


    //mostly duplicated from TetrahedronCell.cpp

    p.new_cell(v0, e3, e4, e5);
    p.new_cell(v1, e1, e2, e5);
    p.new_cell(v2, e0, e2, e4);
    p.new_cell(v3, e0, e1, e3);

    const Point p0 = e[0].midpoint();
    const Point p1 = e[1].midpoint();
    const Point p2 = e[2].midpoint();
    const Point p3 = e[3].midpoint();
    const Point p4 = e[4].midpoint();
    const Point p5 = e[5].midpoint();
    const double d05 = p0.distance(p5);
    const double d14 = p1.distance(p4);
    const double d23 = p2.distance(p3);

    // Then divide the remaining octahedron into 4 tetrahedra
    if (d05 <= d14 && d14 <= d23)
    {
      p.new_cell(e0, e1, e2, e5);
      p.new_cell(e0, e1, e3, e5);
      p.new_cell(e0, e2, e4, e5);
      p.new_cell(e0, e3, e4, e5);
    }
    else if (d14 <= d23)
    {
      p.new_cell(e0, e1, e2, e4);
      p.new_cell(e0, e1, e3, e4);
      p.new_cell(e1, e2, e4, e5);
      p.new_cell(e1, e3, e4, e5);
    }
    else
    {
      p.new_cell(e0, e1, e2, e3);
      p.new_cell(e0, e2, e3, e4);
      p.new_cell(e1, e2, e3, e5);
      p.new_cell(e2, e3, e4, e5);
    }

  }

  std::cout << "Added " << p.cell_topology().size() << " as cell topology" << std::endl;
  std::cout << "Added " << p.vertex_coordinates().size() << " as vertices" << std::endl;

  p.partition(new_mesh);


}

