// Copyright (C) 2007 Kristian B. Oelgaard
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

#ifndef __INTERVAL_MESH_H
#define __INTERVAL_MESH_H

#include <cstddef>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// Interval mesh of the 1D line [a,b].  Given the number of cells
  /// (n) in the axial direction, the total number of intervals will
  /// be n and the total number of vertices will be (n + 1).

  class IntervalMesh : public Mesh
  {
  public:

    /// Factory
    ///
    /// @param    n (std::size_t)
    ///         The number of cells.
    /// @param    x (std::array<double, 2>)
    ///         The end points
    ///
    /// @code{.cpp}
    ///
    ///         // Create a mesh of 25 cells in the interval [-1,1]
    ///         auto mesh = IntervalMesh::create(25, {-1.0, 1.0});
    /// @endcode
    static Mesh create(std::size_t n, std::array<double, 2> x);

    /// Factory
    ///
    /// @param    comm (MPI_Comm)
    ///         MPI communicator
    /// @param    n (std::size_t)
    ///         The number of cells.
    /// @param    x (std::array<double, 2>)
    ///         The end points
    ///
    /// @code{.cpp}
    ///
    ///         // Create a mesh of 25 cells in the interval [-1,1]
    ///         IntervalMesh mesh(MPI_COMM_WORLD, 25, -1.0, 1.0);
    /// @endcode
    static Mesh create(MPI_Comm comm, std::size_t n, std::array<double, 2> x)
    {
      Mesh mesh;
      build(mesh, n, x);
      return mesh;
    }

    /// Constructor
    ///
    /// @param    n (std::size_t)
    ///         The number of cells.
    /// @param    a (double)
    ///         The minimum point (inclusive).
    /// @param    b (double)
    ///         The maximum point (inclusive).
    ///
    /// @code{.cpp}
    ///
    ///         // Create a mesh of 25 cells in the interval [-1,1]
    ///         IntervalMesh mesh(25, -1.0, 1.0);
    /// @endcode
    IntervalMesh(std::size_t n, double a, double b);

    /// Constructor
    ///
    /// @param    comm (MPI_Comm)
    ///         MPI communicator
    /// @param    n (std::size_t)
    ///         The number of cells.
    /// @param    a (double)
    ///         The minimum point (inclusive).
    /// @param    b (double)
    ///         The maximum point (inclusive).
    ///
    /// @code{.cpp}
    ///
    ///         // Create a mesh of 25 cells in the interval [-1,1]
    ///         IntervalMesh mesh(MPI_COMM_WORLD, 25, -1.0, 1.0);
    /// @endcode
    IntervalMesh(MPI_Comm comm, std::size_t n, double a, double b);

  private:

    // Build mesh
    static void build(Mesh& mesh, std::size_t n, std::array<double, 2> x);

  };

}

#endif
