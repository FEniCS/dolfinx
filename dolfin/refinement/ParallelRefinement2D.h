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
// Last Changed: 2012-12-19

#include <boost/unordered_map.hpp>

namespace dolfin 
{
  class Mesh;
  template<typename T> class EdgeFunction;


  class ParallelRefinement2D
  {
  public:

    static void refine(Mesh& new_mesh, const Mesh& mesh, 
                       const MeshFunction<bool>& refinement_marker);
    
    
  private:

    // Reorder vertices into global order - needed for LocalMeshData to re-partition
    static void reorder_vertices_by_global_indices(std::vector<double>& vertex_coords, const std::size_t gdim,
                                            const std::vector<std::size_t>& global_indices);
  
    // Used to find longest edge of a cell
    static bool length_compare(std::pair<double, std::size_t> a, std::pair<double, std::size_t> b);

    // Transmit shared values of an EdgeFunction between processes
    static void update_logical_edgefunction(EdgeFunction<bool>& values, 
                                     const boost::unordered_map<std::size_t, std::size_t>& global_to_local,
                                     const boost::unordered_map<std::size_t, std::size_t>& shared_edges);

    // Calculate which edges should be 'reference' edges for the RGB Carstensen type triangulation
    static void generate_reference_edges(const Mesh& mesh, std::vector<std::size_t>& ref_edge);

    // Work out shared edges - in the future, this will be obtained from MeshConnectivity
    static void get_shared_edges(boost::unordered_map<std::size_t, std::size_t>& shared_edges,
                          boost::unordered_map<std::size_t, std::size_t>& global_to_local,
                          const Mesh &mesh);
  
  };

}
