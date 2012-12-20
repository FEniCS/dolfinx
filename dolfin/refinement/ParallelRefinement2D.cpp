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
// Last Changed: 2012-12-20

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


#include "ParallelRefinement2D.h"

using namespace dolfin;


// This is needed to interface with MeshPartitioning/LocalMeshData, 
// which expects the vertices in global order 
// This is inefficient, and needs to be addressed in MeshPartitioning.cpp
// where they are redistributed again.

void ParallelRefinement2D::reorder_vertices_by_global_indices(std::vector<double>& vertex_coords,
                                                              const std::size_t gdim,
                                                              const std::vector<std::size_t>& global_indices)
{
  Timer t("Parallel Refine: reorder vertices");
  // FIXME: be more efficient with MPI

  dolfin_assert(gdim*global_indices.size() == vertex_coords.size());

  boost::multi_array_ref<double, 2> vertex_array(vertex_coords.data(),
                      boost::extents[vertex_coords.size()/gdim][gdim]);

  // Calculate size of overall global vector by finding max index value
  // anywhere
  const std::size_t global_vector_size
    = MPI::max(*std::max_element(global_indices.begin(), global_indices.end())) + 1;

  // Send unwanted values off process
  const std::size_t num_processes = MPI::num_processes();
  std::vector<std::vector<std::pair<std::size_t, std::vector<double> > > > values_to_send(num_processes);
  std::vector<std::size_t> destinations(num_processes);

  // Set up destination vector for communication with remote processes
  for(std::size_t process_j = 0; process_j < num_processes ; ++process_j)
    destinations[process_j] = process_j;

  // Go through local vector and append value to the appropriate list
  // to send to correct process
  for(std::size_t i = 0; i < vertex_array.shape()[0] ; ++i)
  {
    const std::size_t global_i = global_indices[i];
    const std::size_t process_i = MPI::index_owner(global_i, global_vector_size);
    const std::vector<double> v(vertex_array[i].begin(), vertex_array[i].end());
    values_to_send[process_i].push_back(std::make_pair(global_i, v));
  }

  // Redistribute the values to the appropriate process - including self
  // All values are "in the air" at this point, so local vector can be cleared
  std::vector<std::vector<std::pair<std::size_t,std::vector<double> > > > received_values;
  MPI::distribute(values_to_send, destinations, received_values);

  // When receiving, just go through all received values
  // and place them in the local partition of the global vector.
  const std::pair<std::size_t, std::size_t> range = MPI::local_range(global_vector_size);
  vertex_coords.resize((range.second - range.first)*gdim);
  boost::multi_array_ref<double, 2> new_vertex_array(vertex_coords.data(),
                     boost::extents[range.second - range.first][gdim]);

  for(std::size_t i = 0; i < received_values.size(); ++i)
  {
    const std::vector<std::pair<std::size_t, std::vector<double> > >& received_global_data = received_values[i];
    for(std::size_t j = 0; j < received_global_data.size(); ++j)
    {
      const std::size_t global_i = received_global_data[j].first;
      dolfin_assert(global_i >= range.first && global_i < range.second);
      std::copy(received_global_data[j].second.begin(),
                received_global_data[j].second.end(),
                new_vertex_array[global_i - range.first].begin());

    }
  }
}
//-----------------------------------------------------------------------------

bool ParallelRefinement2D::length_compare(std::pair<double, std::size_t> a, std::pair<double, std::size_t> b)
{
  return (a.first > b.first);
}

//-----------------------------------------------------------------------------

// logical "or" of edgefunction on boundaries
void ParallelRefinement2D::update_logical_edgefunction(EdgeFunction<bool>& values, 
           const boost::unordered_map<std::size_t, std::size_t>& global_to_local,
           const boost::unordered_map<std::size_t, std::size_t>& shared_edges)
{
  Timer t("update logical edgefunction");
  
  uint num_processes = MPI::num_processes();

  // Create a list of edges on this process that are 'true' and copy to remote sharing processes
  std::vector<uint>destinations(num_processes);
  std::vector<std::vector<std::size_t> > values_to_send(num_processes);
  std::vector<std::vector<std::size_t> > received_values;

  for(uint i = 0; i < num_processes; ++i)
    destinations[i] = i;

  for(boost::unordered_map<std::size_t, std::size_t>::const_iterator sh_edge = shared_edges.begin();
      sh_edge != shared_edges.end(); sh_edge++)
  {
    const std::size_t global_index = sh_edge->first;
    //for const map, cannot use global_to_local[global_index]
    const std::size_t local_index = global_to_local.find(global_index)->second; 
    if(values[local_index] == true)
    {
      const std::size_t proc = sh_edge->second;
      values_to_send[proc].push_back(global_index);
    }
  }
  
  MPI::distribute(values_to_send, destinations, received_values);

  // Flatten received values and set EdgeFunction true at each global_index
  // received
  for(std::vector<std::vector<std::size_t> >::iterator r=received_values.begin();
      r != received_values.end(); ++r)
    for(std::vector<std::size_t>::iterator global_index = r->begin(); 
        global_index != r->end(); ++global_index)
    {
      std::size_t local_index = global_to_local.find(*global_index)->second;
      values[local_index] = true;
    }
  
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

//-----------------------------------------------------------------------------

void ParallelRefinement2D::get_shared_edges(boost::unordered_map<std::size_t, std::size_t>& shared_edges,
                                            boost::unordered_map<std::size_t, std::size_t>& global_to_local,
                                            const Mesh &mesh)
{
  
  uint D = mesh.topology().dim();

  // Work out shared edges, and which processes they exist on
  // There are special cases where it is more difficult to determine
  // edge ownership, but these are rare in 2D. Raise an error if
  // it happens.
  // Ultimately, this functionality will be provided inside MeshConnectivity or similar
  
  const std::map<std::size_t, std::set<std::size_t> >& shared_vertices = mesh.topology().shared_entities(0);

  for(EdgeIterator edge(mesh); !edge.end(); ++edge)
  {
    if(edge->num_entities(D) == 1 && edge->num_global_entities(D) == 2)
    {
      global_to_local.insert(std::make_pair(edge->global_index(),edge->index()));

      // This is a shared edge - find sharing process
      // by taking the intersection of the sets of processes of 
      // the two attached vertices (in 2D)
      VertexIterator v(*edge);
      const std::set<std::size_t>& set1 
        = shared_vertices.find(v->global_index())->second;
      ++v;
      const std::set<std::size_t>& set2 
        = shared_vertices.find(v->global_index())->second;
      
      std::vector<std::size_t> result(set1.size() + set2.size());
      uint nprocs = std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), result.begin()) - result.begin();

      if(nprocs == 1)
        shared_edges.insert(std::make_pair(edge->global_index(), result[0]));
      else
        dolfin_error("main.cpp","create shared edges","An unusual condition has occured, due to the mesh partitioning.\nMaybe try again with a different partitioner, or different number of processes.");
    }
  }
  
  std::cout << "n(shared_edges) = " << shared_edges.size() << std::endl;

}


void ParallelRefinement2D::refine(Mesh& new_mesh, const Mesh& mesh, 
                                  const MeshFunction<bool>& refinement_marker)
{
  
  if(MPI::num_processes()==1)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in parallel");
  }

  // Ensure connectivity is there
  uint D = mesh.topology().dim();
  if(D != 2)
  {
    dolfin_error("ParallelRefinement2D.cpp",
                 "refine mesh",
                 "Only works in 2D");
  }

  uint D1 = D - 1;
  mesh.init(D1, D);

  boost::unordered_map<std::size_t, std::size_t> shared_edges;   // global_index => process
  boost::unordered_map<std::size_t, std::size_t> global_to_local; // self-explanatory map for shared edges
  get_shared_edges(shared_edges, global_to_local, mesh);

  // Vector over all cells - the reference edge is the cell's edge (0, 1 or 2) 
  // which always must split, if any edge splits in the cell
  std::vector<std::size_t> ref_edge;
  generate_reference_edges(mesh, ref_edge);
  
  // ***** Output for diagnostics
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
  // *****


  // Set marked edges from marked cells
  EdgeFunction<bool> markedEdges(mesh,false);
  
  // Mark all edges of marked cells
  for(CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if(refinement_marker[*cell])
      for(EdgeIterator edge(*cell); !edge.end(); ++edge)
        markedEdges[*edge] = true;
  }
  
  
  // Mark reference edges of cells with any marked edge
  // and repeat until no more marking takes place

  uint update_count = 1;
  while(update_count != 0)
  {
    update_count = 0;
    
    // Transmit values between processes - could be streamlined
    update_logical_edgefunction(markedEdges, global_to_local, shared_edges);

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
  
  // Diagnostic output
  File markedEdgeFile("marked_edges.xdmf");
  markedEdgeFile << markedEdges;

  // Stage 3a - collect up marked edges which are owned locally...
  // these will provide new vertices.
  // if they are shared, then the new global index needs to be sent off-process

  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();
  std::vector<double> midpoint_coordinates;

  std::vector<std::vector<std::pair<std::size_t, std::size_t> > > values_to_send(num_processes);
  std::vector<std::size_t> destinations(num_processes);
  // Set up destination vector for communication with remote processes
  for(std::size_t process_j = 0; process_j < num_processes ; ++process_j)
    destinations[process_j] = process_j;

  // Mapping from global edge index to new global vertex index
  std::map<std::size_t, std::size_t> global_edge_to_new_vertex;

  // Tally up unshared marked edges, and shared marked edges which are owned on this process.
  // Index them sequentially from zero.

  std::size_t n=0;
  for(EdgeIterator edge(mesh); !edge.end(); ++edge)
  {
    if(markedEdges[*edge])
    {
      const std::size_t global_i = edge->global_index();

      if(shared_edges.count(global_i) == 0) //local new vertex
      {
        midpoint_coordinates.push_back(edge->midpoint()[0]);
        midpoint_coordinates.push_back(edge->midpoint()[1]);
        global_edge_to_new_vertex[global_i] = n++;
      }
      else if(shared_edges.find(global_i)->second > process_number)
      { //local new vertex to be shared with another process
        midpoint_coordinates.push_back(edge->midpoint()[0]);
        midpoint_coordinates.push_back(edge->midpoint()[1]);
        global_edge_to_new_vertex[global_i] = n++;
      } 
      // else new vertex is remotely owned
       
    }
  }

  // Calculate global range for new local vertices
  const uint gdim = mesh.geometry().dim();
  const uint tdim = mesh.topology().dim();
  const std::size_t num_new_vertices = n;
  const std::size_t global_offset = MPI::global_offset(num_new_vertices, true) 
                                  + mesh.size_global(0);

  // Add offset to map, and collect up any shared new vertices that need to send the new index off-process
  for(std::map<std::size_t, std::size_t>::iterator gl_edge = global_edge_to_new_vertex.begin();
      gl_edge != global_edge_to_new_vertex.end(); ++gl_edge)
  {
    gl_edge->second += global_offset; // add global_offset to map, to get new global index of new vertices

    const std::size_t global_i = gl_edge->first;
    if(shared_edges.count(global_i) != 0) //shared, but locally owned. 
    {
      const uint remote_process = shared_edges[global_i];
      values_to_send[remote_process].push_back(std::make_pair(global_i, gl_edge->second));
    }
  }

  // send new vertex indices to remote processes and receive
  std::vector<std::vector<std::pair<std::size_t,std::size_t> > > received_values;
  MPI::distribute(values_to_send, destinations, received_values);

  // Flatten and add received remote global indices to map 
  for(std::vector<std::vector<std::pair<std::size_t, std::size_t> > >::iterator p = received_values.begin();
      p != received_values.end(); ++p)
    for(std::vector<std::pair<std::size_t, std::size_t> >::iterator q = p->begin(); q != p->end(); ++q)
    {
      global_edge_to_new_vertex[q->first] = q->second;
    }
  
  std::cout << "Process:" << process_number << " " << num_new_vertices << " new vertices, "
            << "Offset = " << global_offset
            << std::endl;

  // Now add new vertex coordinates to existing, and index using new global indexing.
  // Reorder so that MeshPartitioning.cpp can find them. After that, we are done with 
  // coordinates, and just need to rebuild the topology.
  
  std::vector<double> vertex_coordinates(mesh.coordinates());
  vertex_coordinates.insert(vertex_coordinates.end(),
                            midpoint_coordinates.begin(),
                            midpoint_coordinates.end());

  std::vector<std::size_t> global_indices(mesh.topology().global_indices(0));
  for(uint i=0; i < num_new_vertices; i++)
  {
    global_indices.push_back(i+global_offset);
  }
  reorder_vertices_by_global_indices(vertex_coordinates, gdim, global_indices);
  
  std::cout << "vertices= " << vertex_coordinates.size() << std::endl;

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
    std::size_t cell_index = cell->index();
    std::size_t ref = ref_edge[cell_index];
    std::vector<std::size_t>rgb_count;

    for(EdgeIterator edge(*cell); !edge.end(); ++edge)
    {
      if(markedEdges[*edge])
        rgb_count.push_back(edge->global_index());
    }

    if(rgb_count.size() == 0) //straight copy of cell - easy
    {
      for(VertexIterator v(*cell); !v.end(); ++v)
        new_cell_topology.push_back(v->global_index());
      new_ref_edge.push_back(ref);
    }
    else if(rgb_count.size() == 1) // "green" refinement
    {
      // Always splitting the reference edge...
      
      EdgeIterator e(*cell);
      VertexIterator v(*cell);
      std::size_t eref = global_edge_to_new_vertex[e[ref].global_index()];
      std::size_t vref = v[ref].global_index();
      std::size_t vnext = v[(ref + 1)%3].global_index();
      std::size_t vlast = v[(ref + 2)%3].global_index();
      
      new_cell_topology.push_back(eref);
      new_cell_topology.push_back(vref);
      new_cell_topology.push_back(vnext);
      new_ref_edge.push_back(ref);      

      new_cell_topology.push_back(eref);
      new_cell_topology.push_back(vlast);
      new_cell_topology.push_back(vref);
      new_ref_edge.push_back(ref);      

    }
    else if(rgb_count.size() == 2) // "blue" refinement
    {
      EdgeIterator e(*cell);
      VertexIterator v(*cell);

      std::size_t vref = ref;
      std::size_t eref = e[vref].global_index();
      std::size_t vnonref, enonref, vother;

      if(eref == rgb_count[0])
        enonref = rgb_count[1];
      else
        enonref = rgb_count[0];

      if(enonref == e[0].global_index())
      {
        vnonref = 0;
      }
      else if(enonref == e[1].global_index())
      {
        vnonref = 1;
      }
      else
      {
        vnonref = 2;
      }

      vother = 3 - vref - vnonref;
      
      vref = v[vref].global_index();
      vnonref = v[vnonref].global_index();
      vother = v[vother].global_index();

      eref = global_edge_to_new_vertex[eref];
      enonref = global_edge_to_new_vertex[enonref];

      new_cell_topology.push_back(eref);
      new_cell_topology.push_back(enonref);
      new_cell_topology.push_back(vref);
      new_ref_edge.push_back(ref_edge[cell_index]);      

      new_cell_topology.push_back(eref);
      new_cell_topology.push_back(vnonref);
      new_cell_topology.push_back(vref);
      new_ref_edge.push_back(ref_edge[cell_index]);      

      new_cell_topology.push_back(eref);
      new_cell_topology.push_back(enonref);
      new_cell_topology.push_back(vother);
      new_ref_edge.push_back(ref_edge[cell_index]);      


    }
    else if(rgb_count.size() == 3) // "red" refinement - all split (1->4) cells
    {
      EdgeIterator e(*cell);
      VertexIterator v(*cell);
      std::size_t e0 = global_edge_to_new_vertex[e[0].global_index()];
      std::size_t e1 = global_edge_to_new_vertex[e[1].global_index()];
      std::size_t e2 = global_edge_to_new_vertex[e[2].global_index()];
      std::size_t v0 = v[0].global_index();
      std::size_t v1 = v[1].global_index();
      std::size_t v2 = v[2].global_index();
      
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

  std::cout << "gdim = " << gdim << std::endl;
  
  std::size_t num_local_cells = new_cell_topology.size()/mesh_data.num_vertices_per_cell;

  std::cout << "Num local cells = " << num_local_cells << std::endl;

  mesh_data.num_global_cells = MPI::sum(num_local_cells);
  std::cout << "Num global cells = " << mesh_data.num_global_cells << std::endl;
  mesh_data.global_cell_indices.resize(num_local_cells);
  std::size_t idx_global_offset = MPI::global_offset(num_local_cells, true);
  for(std::size_t i = 0; i < num_local_cells ; i++)
    mesh_data.global_cell_indices[i] = idx_global_offset + i;
  
  mesh_data.cell_vertices.resize(boost::extents[num_local_cells][mesh_data.num_vertices_per_cell]);
  std::copy(new_cell_topology.begin(),new_cell_topology.end(),mesh_data.cell_vertices.data());

  std::size_t num_local_vertices = vertex_coordinates.size()/gdim;
  mesh_data.num_global_vertices = MPI::sum(num_local_vertices);
  std::cout << "Num global vertices = " << mesh_data.num_global_vertices << std::endl;

  mesh_data.vertex_coordinates.resize(boost::extents[num_local_vertices][gdim]);
  std::copy(vertex_coordinates.begin(), vertex_coordinates.end(), mesh_data.vertex_coordinates.data());
  mesh_data.vertex_indices.resize(num_local_vertices);

  std::pair<std::size_t, std::size_t> local_range = MPI::local_range(mesh_data.num_global_vertices, true);
  for(std::size_t i = 0; i < num_local_vertices ; i++)
    mesh_data.vertex_indices[i] = local_range.first + i;

  //  Mesh new_mesh;
  MeshPartitioning::build_distributed_mesh(new_mesh, mesh_data);

  CellFunction<std::size_t> partitioning1(mesh, process_number);
  CellFunction<std::size_t> partitioning2(new_mesh, process_number);

  File meshFile1("old_mesh.xdmf");
  meshFile1 << partitioning1;  

  File meshFile2("new_mesh.xdmf");
  meshFile2 << partitioning2;  

  //  HDF5File H5mesh("mesh.h5","w");
  //  H5mesh.write(new_mesh,"new_mesh");

  // new_ref_edge exists, but needs reordering...

}

