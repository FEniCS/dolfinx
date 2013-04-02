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
// Last Changed: 2013-01-03

namespace dolfin 
{
  class Mesh;

  class ParallelRefinement2D
  {
  public:

    // refine with marker
    static void refine(Mesh& new_mesh, const Mesh& mesh, 
                       const MeshFunction<bool>& refinement_marker);
    
    // uniform refine
    static void refine(Mesh& new_mesh, const Mesh& mesh);
    
  private:

    // Used to find longest edge of a cell, when working out reference edges
    static bool length_compare(std::pair<double, std::size_t> a, std::pair<double, std::size_t> b);

    // Calculate which edges should be 'reference' edges for the RGB Carstensen type triangulation
    static void generate_reference_edges(const Mesh& mesh, std::vector<std::size_t>& ref_edge);
  
  };

}
