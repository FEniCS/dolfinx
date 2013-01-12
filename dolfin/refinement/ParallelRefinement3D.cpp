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
// Last Changed: 2013-01-12

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
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/LocalMeshData.h>

#include <dolfin/refinement/ParallelRefinement.h>

#include "ParallelRefinement3D.h"

using namespace dolfin;


void ParallelRefinement3D::refine(Mesh& new_mesh, const Mesh& mesh,
                                  const MeshFunction<bool>& refinement_marker)
{
  std::size_t tdim = mesh.topology().dim();
  
  warning("not working yet");

  // Ensure connectivity from cells to edges
  mesh.init(1);
  mesh.init(1, tdim);

  ParallelRefinement p(mesh);
  
  EdgeFunction<bool> markedEdges(mesh, false);

  // Mark all edges of marked cells
  
  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if(refinement_marker[*cell])
    {
      for (EdgeIterator edge(*cell); !edge.end(); ++edge)
      {
        markedEdges[*edge] = true;
      }
    }
  }
  
  std::size_t update_count = 1;
  
  while (update_count != 0)
  {
    // Transmit shared marked edges
    p.update_logical_edgefunction(markedEdges);

    update_count = 0;
    
    for(CellIterator cell(mesh); !cell.end(); ++cell)
    {
      std::size_t n_marked = 0;
      for (EdgeIterator edge(*cell); !edge.end(); ++edge)
      {
        if(markedEdges[*edge]) 
          n_marked++;
      }
      
      if (n_marked > 3)
      { // mark all, if more than 3 edges are already marked 
        for (EdgeIterator edge(*cell); !edge.end(); ++edge)
          markedEdges[*edge] = true;
        update_count = 1;
      }
      
      if (n_marked == 3)
      {
        // With 3 marked edges, they must be all on the same face, otherwise, just mark all
        std::size_t nmax=0;
        for (FaceIterator face(*cell); !face.end(); ++face)
        {
          std::size_t n=0;
          for(EdgeIterator edge(*face); !edge.end(); ++edge)
          {
            if(markedEdges[*edge])
              n++;
          }
          nmax = (n > nmax) ? n : nmax;
        }        
        if(nmax != 3)
        {
          for (EdgeIterator edge(*cell); !edge.end(); ++edge)
            markedEdges[*edge] = true;
          update_count = 1;
        }
      }

    }

    update_count = MPI::sum(update_count);
    
  }
  
  // All cells now have either 0, 1, 2, 3* or 6 edges marked.
  // * (3 are all on the same face)

  // Create new vertices
  p.create_new_vertices(markedEdges);

  std::map<std::size_t, std::size_t>& edge_to_new_vertex = p.edge_to_new_vertex();

  // Create new topology

  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {    
    EdgeIterator e(*cell);
    VertexIterator v(*cell);

    std::vector<std::size_t> rgb;
    for(std::size_t j = 0 ; j < 6 ; ++j)
    {
      if(markedEdges[e[j]])
        rgb.push_back(j);
    }


    if(rgb.size() == 0) //straight copy of cell (1->1)
    {
      const std::size_t v0 = v[0].global_index();
      const std::size_t v1 = v[1].global_index();
      const std::size_t v2 = v[2].global_index();
      const std::size_t v3 = v[3].global_index();
      p.new_cell(v0, v1, v2, v3);
    }
    else if(rgb.size() == 1) // "green" refinement (bisection)
    {
      const std::size_t new_edge = rgb[0];
      const std::size_t v_new = edge_to_new_vertex[e[new_edge].index()];
      VertexIterator vn(e[new_edge]);
      const std::size_t v_near_0 = vn[0].global_index();
      const std::size_t v_near_1 = vn[1].global_index();
      // opposite edges always add up to 5
      const std::size_t opp_edge = 5 - new_edge;
      VertexIterator vf(e[opp_edge]);
      const std::size_t v_far_0 = vf[0].global_index();
      const std::size_t v_far_1 = vf[1].global_index();

      p.new_cell(v_far_0, v_far_1, v_new, v_near_0);
      p.new_cell(v_far_0, v_far_1, v_new, v_near_1);
    }
    else if(rgb.size() == 2) 
    {
      const std::size_t new_edge_0 = rgb[0];
      const std::size_t new_edge_1 = rgb[1];
      const std::size_t e0 = edge_to_new_vertex[e[new_edge_0].index()];
      const std::size_t e1 = edge_to_new_vertex[e[new_edge_1].index()];
      VertexIterator v0(e[new_edge_0]);
      VertexIterator v1(e[new_edge_1]);

      std::size_t v_common(0), v_leg_0(0), v_leg_1(0);      
      for (std::size_t i = 0; i < 2; ++i)
      {
        for (std::size_t j = 0; j < 2; ++j)
        {
          if (v0[i] == v1[j])
          {
            v_common = v0[i].index();
            v_leg_0 = v0[1 - i].index();
            v_leg_1 = v1[1 - j].index();
          }
        }
      }
      
      // need to find the 'uncommon' vertex of the two edges
      // which is furthest from both
      std::size_t v_far = 0;
      
      for(std::size_t i = 0; i < 4; ++i)
      {
        const std::size_t v_i = v[i].index();
        if(v_i != v_common && v_i != v_leg_0 && v_i != v_leg_1)
        {
          v_far = v_i;
        }
      }
      
      // find distance across trapezoid, and choose shortest, if possible
      const Point p_leg_0 = Vertex(mesh, v_leg_0).point();
      const Point p_leg_1 = Vertex(mesh, v_leg_1).point();
      const double d0 = p_leg_0.distance(e[new_edge_1].midpoint());
      const double d1 = p_leg_1.distance(e[new_edge_0].midpoint());

      p.new_cell(v_far, v_common, e0, e1);
      
      if(d0 == d1) // problem case - use global index to determine precedence
      {
        if(Vertex(mesh, v_leg_0).global_index() 
           > Vertex(mesh, v_leg_1).global_index())
        {
          p.new_cell(v_far, e0, e1, v_leg_1);
          p.new_cell(v_far, e0, v_leg_0, v_leg_1);
        }
        else 
        {
          p.new_cell(v_far, e1, e0, v_leg_0);
          p.new_cell(v_far, e1, v_leg_1, v_leg_0);
        }
      } 
      else if(d0 > d1) // vleg0->e1 is longer
      {
        p.new_cell(v_far, e0, e1, v_leg_1);
        p.new_cell(v_far, e0, v_leg_0, v_leg_1);
      }
      else 
      {
        p.new_cell(v_far, e1, e0, v_leg_0);
        p.new_cell(v_far, e1, v_leg_1, v_leg_0);
      }
    }
    else if(rgb.size() == 3) // refinement of one face into 4 triangles
    {
        dolfin_error("ParallelRefinement3D.cpp",
                     "refine",
                     "Error in making new cells");
    }
    else if(rgb.size() == 6)
    {
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
  }
  
  p.partition(new_mesh);

}

  
 


// Uniform refinement

void ParallelRefinement3D::refine(Mesh& new_mesh, const Mesh& mesh)
{
  if(MPI::num_processes()==1)
  {
    dolfin_error("ParallelRefinement3D.cpp",
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

  p.partition(new_mesh);

}

// std::vector<std::size_t> ParallelRefinement3D::edge_vertex_one(std::size_t e1)
// {
//   // Work out opposite and neighbouring vertices of an edge
// }


  

// void ParallelRefinement3D::vertex_edge_pair(std::size_t e1, std::size_t e2)
// {
//   // Work out the opposite vertex and the shared vertex for two edges in a tetrahedron
//   static unsigned  char tdata[] = {0x03, 0x05, 0x09, 0x06, 0x0A, 0x0C};
//   static char rmap[] = {-1, 0, 1, -1, 2, -1, -1, -1, 3};
  
//   std::size_t v_idx = (tdata[e1] & tdata[e2]);
  
//   std::size_t v_opp = rmap[v_idx];
  
//   v_idx = (~tdata[e1] & ~tdata[e2])&0x0F;
  
//   std::size_t v_shared = rmap[v_idx];
  
// }

