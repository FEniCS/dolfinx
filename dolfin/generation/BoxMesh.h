// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
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

#ifndef __BOX_H
#define __BOX_H

#include <array>
#include <cstddef>
#include <dolfin/common/MPI.h>
#include <dolfin/mesh/Mesh.h>

namespace dolfin
{

  /// Tetrahedral mesh of the 3D rectangular prism spanned by two
  /// points p0 and p1. Given the number of cells (nx, ny, nz) in each
  /// direction, the total number of tetrahedra will be 6*nx*ny*nz and
  /// the total number of vertices will be (nx + 1)*(ny + 1)*(nz + 1).

  class BoxMesh : public Mesh
  {
  public:

    /// Create a uniform finite element _Mesh_ over the rectangular
    /// prism spanned by the two _Point_s p0 and p1. The order of the
    /// two points is not important in terms of minimum and maximum
    /// coordinates.
    ///
    /// @param p (std::array<_Point_, 2>)
    ///         Points of box.
    /// @param n (std::array<double, 3> )
    ///         Number of cells in each direction.
    ///
    /// @code{.cpp}
    ///         // Mesh with 8 cells in each direction on the
    ///         // set [-1,2] x [-1,2] x [-1,2].
    ///         Point p0(-1, -1, -1);
    ///         Point p1(2, 2, 2);
    ///         auto mesh = BoxMesh::create({p0, p1}, {8, 8, 8});
    /// @endcode
    static Mesh create(const std::array<Point,2 >& p, std::array<std::size_t, 3> n)
    { return create(MPI_COMM_WORLD, p, n); }

    /// Create a uniform finite element _Mesh_ over the rectangular
    /// prism spanned by the two _Point_s p0 and p1. The order of the
    /// two points is not important in terms of minimum and maximum
    /// coordinates.
    ///
    /// @param comm (MPI_Comm)
    ///         MPI communicator
    /// @param p (std::array<_Point_, 2>)
    ///         Points of box.
    /// @param n (std::array<double, 3> )
    ///         Number of cells in each direction.
    ///
    /// @code{.cpp}
    ///         // Mesh with 8 cells in each direction on the
    ///         // set [-1,2] x [-1,2] x [-1,2].
    ///         Point p0(-1, -1, -1);
    ///         Point p1(2, 2, 2);
    ///         auto mesh = BoxMesh::create({p0, p1}, {8, 8, 8});
    /// @endcode
    static Mesh create(MPI_Comm comm, const std::array<Point, 2>& p,
                       std::array<std::size_t, 3> n)
    {
      Mesh mesh(comm);
      build(mesh, p, n);
      return mesh;
    }

    /// Deprecated
    /// Create a uniform finite element _Mesh_ over the rectangular
    /// prism spanned by the two _Point_s p0 and p1. The order of the
    /// two points is not important in terms of minimum and maximum
    /// coordinates.
    ///
    /// @param p0 (_Point_)
    ///         First point.
    /// @param p1 (_Point_)
    ///         Second point.
    /// @param nx (double)
    ///         Number of cells in x-direction.
    /// @param ny (double)
    ///         Number of cells in y-direction.
    /// @param nz (double)
    ///         Number of cells in z-direction.
    ///
    /// @code{.cpp}
    ///         // Mesh with 8 cells in each direction on the
    ///         // set [-1,2] x [-1,2] x [-1,2].
    ///         Point p0(-1, -1, -1);
    ///         Point p1(2, 2, 2);
    ///         BoxMesh mesh(p0, p1, 8, 8, 8);
    /// @endcode
    BoxMesh(const Point& p0, const Point& p1,
            std::size_t nx, std::size_t ny, std::size_t nz);

    /// Deprecated
    /// Create a uniform finite element _Mesh_ over the rectangular
    /// prism spanned by the two _Point_s p0 and p1. The order of the
    /// two points is not important in terms of minimum and maximum
    /// coordinates.
    ///
    /// @param comm (MPI_Comm)
    ///         MPI communicator
    /// @param p0 (_Point_)
    ///         First point.
    /// @param p1 (_Point_)
    ///         Second point.
    /// @param nx (double)
    ///         Number of cells in x-direction.
    /// @param ny (double)
    ///         Number of cells in y-direction.
    /// @param nz (double)
    ///         Number of cells in z-direction.
    ///
    /// @code{.cpp}
    ///         // Mesh with 8 cells in each direction on the
    ///         // set [-1,2] x [-1,2] x [-1,2].
    ///         Point p0(-1, -1, -1);
    ///         Point p1(2, 2, 2);
    ///         BoxMesh mesh(MPI_COMM_WORLD, p0, p1, 8, 8, 8);
    /// @endcode
    BoxMesh(MPI_Comm comm, const Point& p0, const Point& p1,
            std::size_t nx, std::size_t ny, std::size_t nz);

  private:

    // Build mesh
    static void build(Mesh& mesh, const std::array<Point, 2>& p,
                      std::array<std::size_t, 3> n);

  };

}

#endif
