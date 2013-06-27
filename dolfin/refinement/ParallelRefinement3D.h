// Copyright (C) 2013 Chris Richardson
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
// Last Changed: 2013-01-17


namespace dolfin
{
  class Mesh;
  class Edge;
  class Cell;
  class ParallelRefinement;

  /// Simple class to perform uniform refinement in 3D in parallel

  class ParallelRefinement3D
  {
  public:

    // Uniform refine
    static void refine(Mesh& new_mesh, const Mesh& mesh, bool redistribute);

    // Refine with markers
    static void refine(Mesh& new_mesh, const Mesh& mesh,
                       const MeshFunction<bool>& refinement_marker,
                       bool redistribute);

  private:

    // Full refinement of a tetrahedral cell
    static void eightfold_division(const Cell& cell, ParallelRefinement& p);

    // Work out vertices which are shared by both, one or neither edge
    static std::vector<std::size_t> common_vertices(const Cell& cell,
                                                    const std::size_t edge0,
                                                    const std::size_t edge1);
   };

}
