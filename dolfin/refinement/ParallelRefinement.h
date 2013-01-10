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
// First Added: 2013-01-02
// Last Changed: 2013-01-10

#include <boost/unordered_map.hpp>

namespace dolfin 
{
  class Mesh;
  template<typename T> class EdgeFunction;

  class ParallelRefinement
  {
  public:
    // Create any useful parallel data about the mesh (e.g. shared edges) and store
    ParallelRefinement(const Mesh& mesh);
    ~ParallelRefinement();

    // Experimental management of edge marking
    void mark_edge(std::size_t edge_index);

    // Transfer marked edges between processes
    void update_logical_edgefunction(EdgeFunction<bool>& values);

    // Add new vertex for each marked edge, 
    // and create new_vertex_coordinates and global_edge->new_vertex mapping.
    // Communicate new vertices with MPI to all affected processes.
    void create_new_vertices(const EdgeFunction<bool>& markedEdges);

    // Mapping of old edge (to be removed) to new global vertex number.
    // Useful for forming new topology
    std::map<std::size_t, std::size_t>& edge_to_new_vertex();

    // New vertex coordinates after adding vertices given by marked edges.
    std::vector<double>& vertex_coordinates();
    
    // Add a new cell to the list in 3D or 2D
    void new_cell(std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3); 
    void new_cell(std::size_t i0, std::size_t i1, std::size_t i2); 

    // Get new cell topology as created by new_cell() above
    std::vector<std::size_t>& cell_topology();
    
  private:
    
    // shared edges between processes. In 2D, vector size is 1
    boost::unordered_map<std::size_t, std::vector<std::pair<std::size_t, std::size_t> > > shared_edges;

    // mapping from old local edge index to new global vertex, needed to create new topology
    std::map<std::size_t, std::size_t> local_edge_to_new_vertex;

    // new storage for all coordinates when creating new vertices
    std::vector<double> new_vertex_coordinates;

    // new storage for all cells when creating new topology
    std::vector<std::size_t> new_cell_topology;
    
    // experimental management of marked edges
    std::vector<bool> marked_edges;

    // Mesh reference
    const Mesh& _mesh;

    // Reorder vertices into global order for partitioning
    void reorder_vertices_by_global_indices(std::vector<double>& vertex_coords,
                                            const std::size_t gdim,
                                            const std::vector<std::size_t>& global_indices);
    

  };

}
