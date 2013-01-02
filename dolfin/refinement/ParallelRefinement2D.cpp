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
// Last Changed: 2013-01-02

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

#include "ParallelRefinement2D.h"

using namespace dolfin;

bool ParallelRefinement2D::length_compare(std::pair<double, std::size_t> a, std::pair<double, std::size_t> b)
{
  return (a.first > b.first);
}

//-----------------------------------------------------------------------------
// Work out which edge will be the reference edge for each cell

void ParallelRefinement2D::generate_reference_edges(const Mesh& mesh, std::vector<std::size_t>& ref_edge)
{
  uint D = mesh.topology().dim();
  
  ref_edge.resize(mesh.size(D));
  
  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {
    std::size_t cell_index = cell->index();
    
    std::vector<std::pair<double,std::size_t> > lengths;
    EdgeIterator celledge(*cell);
    for(std::size_t i = 0; i < 3; i++)
    {
      lengths.push_back(std::make_pair(celledge[i].length(), i));
    }
      
    std::sort(lengths.begin(), lengths.end(), length_compare);
    
    // for now - just pick longest edge - this is not the Carstensen algorithm, which tries
    // to pair edges off. Because that is more difficult in parallel, it is not implemented yet.
    const std::size_t edge_index = lengths[0].second;
    ref_edge[cell_index] = edge_index;
  }
}

void ParallelRefinement2D::refine(Mesh& new_mesh, const Mesh& mesh)
{
  if(MPI::num_processes()==1)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in parallel");
  }

  const uint tdim = mesh.topology().dim();
  const uint gdim = mesh.geometry().dim();

  if(tdim != 2 || gdim != 2)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in 2D");
  }

  // Ensure connectivity is there
  mesh.init(tdim - 1, tdim);

  // Create a class to hold most of the refinement information
  ParallelRefinement p(mesh);
  
  // Mark all edges, and create new vertices
  EdgeFunction<bool> markedEdges(mesh, true);
  p.create_new_vertices(markedEdges);
  std::map<std::size_t, std::size_t>& global_edge_to_new_vertex = p.global_edge_to_new_vertex();
  
  // Generate new topology
  std::vector<std::size_t> new_cell_topology;

  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {
    EdgeIterator e(*cell);
    VertexIterator v(*cell);

    const std::size_t v0 = v[0].global_index();
    const std::size_t v1 = v[1].global_index();
    const std::size_t v2 = v[2].global_index();
    const std::size_t e0 = global_edge_to_new_vertex[e[0].global_index()];
    const std::size_t e1 = global_edge_to_new_vertex[e[1].global_index()];
    const std::size_t e2 = global_edge_to_new_vertex[e[2].global_index()];

    new_cell_topology.push_back(v0);
    new_cell_topology.push_back(e2);
    new_cell_topology.push_back(e1);
    
    new_cell_topology.push_back(e2);
    new_cell_topology.push_back(v1);
    new_cell_topology.push_back(e0);

    new_cell_topology.push_back(e1);
    new_cell_topology.push_back(e0);
    new_cell_topology.push_back(v2);

    new_cell_topology.push_back(e0);
    new_cell_topology.push_back(e1);
    new_cell_topology.push_back(e2);    
  }

  LocalMeshData mesh_data;
  mesh_data.num_vertices_per_cell = tdim + 1;
  mesh_data.tdim = tdim;
  mesh_data.gdim = gdim;

  // Copy data to LocalMeshData structures

  const std::size_t num_local_cells = new_cell_topology.size()/mesh_data.num_vertices_per_cell;
  mesh_data.num_global_cells = MPI::sum(num_local_cells);
  mesh_data.global_cell_indices.resize(num_local_cells);
  const std::size_t idx_global_offset = MPI::global_offset(num_local_cells, true);
  for(std::size_t i = 0; i < num_local_cells ; i++)
    mesh_data.global_cell_indices[i] = idx_global_offset + i;
  
  mesh_data.cell_vertices.resize(boost::extents[num_local_cells][mesh_data.num_vertices_per_cell]);
  std::copy(new_cell_topology.begin(),new_cell_topology.end(),mesh_data.cell_vertices.data());

  const std::size_t num_local_vertices = p.vertex_coordinates().size()/gdim;
  mesh_data.num_global_vertices = MPI::sum(num_local_vertices);
  mesh_data.vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
  std::copy(p.vertex_coordinates().begin(), p.vertex_coordinates().end(), mesh_data.vertex_coordinates.data());
  mesh_data.vertex_indices.resize(num_local_vertices);

  const std::size_t vertex_global_offset = MPI::global_offset(num_local_vertices, true);
  for(std::size_t i = 0; i < num_local_vertices ; i++)
    mesh_data.vertex_indices[i] = vertex_global_offset + i;

  MeshPartitioning::build_distributed_mesh(new_mesh, mesh_data);

}

void ParallelRefinement2D::refine(Mesh& new_mesh, const Mesh& mesh, 
                                  const MeshFunction<bool>& refinement_marker)
{
  const uint tdim = mesh.topology().dim();
  const uint gdim = mesh.geometry().dim();

  bool diag=false;   // Enable output for diagnostics
  
  if(MPI::num_processes()==1)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in parallel");
  }

  if(tdim != 2 || gdim != 2)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in 2D");
  }

  // Ensure connectivity is there
  mesh.init(tdim - 1, tdim);

  // Create a class to hold most of the refinement information
  ParallelRefinement p(mesh);

  // Vector over all cells - the reference edge is the cell's edge (0, 1 or 2) 
  // which always must split, if any edge splits in the cell
  std::vector<std::size_t> ref_edge;
  generate_reference_edges(mesh, ref_edge);
   
  if(diag)
  {
    EdgeFunction<bool> ref_edge_fn(mesh,false);
    CellFunction<std::size_t> ref_edge_fn2(mesh);
    for(CellIterator cell(mesh); !cell.end(); ++cell)
    {
      EdgeIterator e(*cell);
      ref_edge_fn[ e[ref_edge[cell->index()]] ] = true;
      ref_edge_fn2[*cell] = ref_edge[cell->index()];
    }
    
    File refEdgeFile("ref_edge.xdmf");
    refEdgeFile << ref_edge_fn;
    
    File refEdgeFile2("ref_edge2.xdmf");
    refEdgeFile2 << ref_edge_fn2;
  }
  
  // Set marked edges from marked cells
  EdgeFunction<bool> markedEdges(mesh,false);
  
  // Mark all edges of marked cells
  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if(refinement_marker[*cell])
      for(EdgeIterator edge(*cell); !edge.end(); ++edge)
      {
        //        p.mark_edge(edge->index());
        markedEdges[*edge] = true;
      }
  }
  
  // Mark reference edges of cells with any marked edge
  // and repeat until no more marking takes place

  uint update_count = 1;
  while(update_count != 0)
  {
    update_count = 0;
    
    // Transmit values between processes - could be streamlined
    p.update_logical_edgefunction(markedEdges);
    
    for(CellIterator cell(mesh); !cell.end(); ++cell)
    {
      bool marked = false;
      // Check if any edge of this cell is marked
      for(EdgeIterator edge(*cell); !edge.end(); ++edge)
      {
        if(markedEdges[*edge])
          marked = true;
      }

      EdgeIterator edge(*cell);
      std::size_t ref_edge_index = edge[ref_edge[cell->index()]].index();

      if(marked && markedEdges[ref_edge_index] == false)
      {
        update_count = 1;
        markedEdges[ref_edge_index] = true;
      }
    }

    std::cout << MPI::process_number() << ":" << update_count << std::endl;
    update_count = MPI::sum(update_count);
  
  }
  
  if(diag)
  {
      // Diagnostic output
    File markedEdgeFile("marked_edges.xdmf");
    markedEdgeFile << markedEdges;
  }

  // Generate new vertices from marked edges, and assign global indices.
  // Also, create mapping from the old global edge index to the new vertex index.
  p.create_new_vertices(markedEdges);
  std::map<std::size_t, std::size_t>& global_edge_to_new_vertex = p.global_edge_to_new_vertex();

  // Stage 4 - do refinement - keeping reference edges somehow?...

  std::vector<std::size_t> new_cell_topology;
  std::vector<std::size_t> new_ref_edge;

  // *******************
  // N.B. although this works after a fashion, it is not right,
  // and needs careful revision - particularly for assignment of the reference edges
  // which are not carried forward, yet. 
  // *******************

  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {
    
    std::size_t rgb_count = 0;
    for(EdgeIterator edge(*cell); !edge.end(); ++edge)
    {
      if(markedEdges[*edge])
        rgb_count++;
    }

    EdgeIterator e(*cell);
    VertexIterator v(*cell);

    const std::size_t ref = ref_edge[cell->index()];
    const std::size_t i0 = ref;
    const std::size_t i1 = (ref + 1)%3;
    const std::size_t i2 = (ref + 2)%3;
    const std::size_t v0 = v[i0].global_index();
    const std::size_t v1 = v[i1].global_index();
    const std::size_t v2 = v[i2].global_index();
    const std::size_t e0 = global_edge_to_new_vertex[e[i0].global_index()];
    const std::size_t e1 = global_edge_to_new_vertex[e[i1].global_index()];
    const std::size_t e2 = global_edge_to_new_vertex[e[i2].global_index()];

    if(rgb_count == 0) //straight copy of cell (1->1)
    {
      for(VertexIterator v(*cell); !v.end(); ++v)
        new_cell_topology.push_back(v->global_index());
      new_ref_edge.push_back(ref);
    }
    else if(rgb_count == 1) // "green" refinement (1->2)
    {
      // Always splitting the reference edge...
      
      new_cell_topology.push_back(e0);
      new_cell_topology.push_back(v0);
      new_cell_topology.push_back(v1);
      new_ref_edge.push_back(ref);      

      new_cell_topology.push_back(e0);
      new_cell_topology.push_back(v2);
      new_cell_topology.push_back(v0);
      new_ref_edge.push_back(ref);      

    }
    else if(rgb_count == 2) // "blue" refinement (1->3) left or right
    {
      
      if(markedEdges[e[i2]])
      {
        new_cell_topology.push_back(e2);
        new_cell_topology.push_back(v1);
        new_cell_topology.push_back(e0);
        new_ref_edge.push_back(ref);              

        new_cell_topology.push_back(e2);
        new_cell_topology.push_back(e0);
        new_cell_topology.push_back(v0);
        new_ref_edge.push_back(ref);              

        new_cell_topology.push_back(e0);
        new_cell_topology.push_back(v2);
        new_cell_topology.push_back(v0);
        new_ref_edge.push_back(ref);              
        
      }
      else if(markedEdges[e[i1]])
      {
        new_cell_topology.push_back(e0);
        new_cell_topology.push_back(v0);
        new_cell_topology.push_back(v1);
        new_ref_edge.push_back(ref);              

        new_cell_topology.push_back(e1);
        new_cell_topology.push_back(e0);
        new_cell_topology.push_back(v2);
        new_ref_edge.push_back(ref);              

        new_cell_topology.push_back(e1);
        new_cell_topology.push_back(v0);
        new_cell_topology.push_back(e0);
        new_ref_edge.push_back(ref);              

      }

    }
    else if(rgb_count == 3) // "red" refinement - all split (1->4) cells
    {
      new_cell_topology.push_back(v0);
      new_cell_topology.push_back(e2);
      new_cell_topology.push_back(e1);
      new_ref_edge.push_back(ref);      

      new_cell_topology.push_back(e2);
      new_cell_topology.push_back(v1);
      new_cell_topology.push_back(e0);
      new_ref_edge.push_back(ref);

      new_cell_topology.push_back(e1);
      new_cell_topology.push_back(e0);
      new_cell_topology.push_back(v2);
      new_ref_edge.push_back(ref);

      new_cell_topology.push_back(e0);
      new_cell_topology.push_back(e1);
      new_cell_topology.push_back(e2);
      new_ref_edge.push_back(ref);
    }
    
  }

  LocalMeshData mesh_data;
  mesh_data.num_vertices_per_cell = tdim + 1;
  mesh_data.tdim = tdim;
  mesh_data.gdim = gdim;

  // Copy data to LocalMeshData structures

  const std::size_t num_local_cells = new_cell_topology.size()/mesh_data.num_vertices_per_cell;
  mesh_data.num_global_cells = MPI::sum(num_local_cells);
  mesh_data.global_cell_indices.resize(num_local_cells);
  const std::size_t idx_global_offset = MPI::global_offset(num_local_cells, true);
  for(std::size_t i = 0; i < num_local_cells ; i++)
    mesh_data.global_cell_indices[i] = idx_global_offset + i;
  
  mesh_data.cell_vertices.resize(boost::extents[num_local_cells][mesh_data.num_vertices_per_cell]);
  std::copy(new_cell_topology.begin(),new_cell_topology.end(),mesh_data.cell_vertices.data());

  const std::size_t num_local_vertices = p.vertex_coordinates().size()/gdim;
  mesh_data.num_global_vertices = MPI::sum(num_local_vertices);
  mesh_data.vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
  std::copy(p.vertex_coordinates().begin(), p.vertex_coordinates().end(), mesh_data.vertex_coordinates.data());
  mesh_data.vertex_indices.resize(num_local_vertices);

  const std::size_t vertex_global_offset = MPI::global_offset(num_local_vertices, true);
  for(std::size_t i = 0; i < num_local_vertices ; i++)
    mesh_data.vertex_indices[i] = vertex_global_offset + i;

  MeshPartitioning::build_distributed_mesh(new_mesh, mesh_data);

  if(diag)
  {
    const std::size_t process_number = MPI::process_number();
    CellFunction<std::size_t> partitioning1(mesh, process_number);
    CellFunction<std::size_t> partitioning2(new_mesh, process_number);
    
    File meshFile1("old_mesh.xdmf");
    meshFile1 << partitioning1;  
    
    File meshFile2("new_mesh.xdmf");
    meshFile2 << partitioning2;  
  }
  
  // new_ref_edge exists, but needs reordering...

}

