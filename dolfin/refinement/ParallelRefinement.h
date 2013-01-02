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
// Last Changed: 2013-01-02

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
    
    void mark_edge(std::size_t edge_index);

    // Transfer marked edges between processes
    void update_logical_edgefunction(EdgeFunction<bool>& values);

    // Add new vertex for each marked edge, and create new_vertex_coordinates and global_edge->new_vertex mapping
    void create_new_vertices(const EdgeFunction<bool>& markedEdges);

    // Mapping of global to local edges
    //    boost::unordered_map<std::size_t, std::size_t>& global_to_local();

    // Shared edges between processes, map from edge to process number
    //    boost::unordered_map<std::size_t, std::size_t>& shared_edges();

    // Mapping of old global edge (to be removed) to new vertex number.
    std::map<std::size_t, std::size_t>& global_edge_to_new_vertex();

    // New vertex coordinates after adding vertices given by marked edges.
    std::vector<double>& vertex_coordinates();
    
  private:

    boost::unordered_map<std::size_t, std::size_t> _global_to_local;
    boost::unordered_map<std::size_t, std::set<std::size_t> > _shared_edges;
    std::map<std::size_t, std::size_t> _global_edge_to_new_vertex;
    std::vector<double> new_vertex_coordinates;
    std::vector<bool> marked_edges;
    bool need_to_transfer;

    const Mesh& _mesh;

    void get_shared_edges();
    void reorder_vertices_by_global_indices(std::vector<double>& vertex_coords,
                                            const std::size_t gdim,
                                            const std::vector<std::size_t>& global_indices);
    

  };

}
