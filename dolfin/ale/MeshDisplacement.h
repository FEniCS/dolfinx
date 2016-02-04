// Copyright (C) 2013 Jan Blechta
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
// First added:  2013-03-05
// Last changed: 2013-03-05

#ifndef __MESH_DISPLACEMENT_H
#define __MESH_DISPLACEMENT_H

#include <memory>
#include <vector>
#include <ufc.h>
#include <dolfin/common/Array.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>

namespace dolfin
{
  class Mesh;

  /// This class encapsulates the CG1 representation of the
  /// displacement of a mesh as an Expression. This is particularly
  /// useful for the displacement returned by mesh smoothers which can
  /// subsequently be used in evaluating forms. The value rank is 1
  /// and the value shape is equal to the geometric dimension of the
  /// mesh.

  class MeshDisplacement : public Expression
  {
  public:

    /// Create MeshDisplacement of given mesh
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         Mesh to be displacement defined on.
    explicit MeshDisplacement(std::shared_ptr<const Mesh> mesh);

    /// Copy constructor
    ///
    /// *Arguments*
    ///     mesh_displacement (_MeshDisplacement_)
    ///         Object to be copied.
    MeshDisplacement(const MeshDisplacement& mesh_displacement);

    /// Destructor
    virtual ~MeshDisplacement();

    /// Extract subfunction
    /// In python available as MeshDisplacement.sub(i)
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         Index of subfunction.
    Function& operator[] (const std::size_t i);

    /// Extract subfunction. Const version
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         Index of subfunction.
    const Function& operator[] (const std::size_t i) const;

    /// Evaluate at given point in given cell.
    ///
    /// *Arguments*
    ///     values (_Array_ <double>)
    ///         The values at the point.
    ///     x (_Array_ <double>)
    ///         The coordinates of the point.
    ///     cell (ufc::cell)
    ///         The cell which contains the given point.
    virtual void eval(Array<double>& values,
		      const Array<double>& x,
                      const ufc::cell& cell) const;

    /// Compute values at all mesh vertices.
    ///
    /// *Arguments*
    ///     vertex_values (_Array_ <double>)
    ///         The values at all vertices.
    ///     mesh (_Mesh_)
    ///         The mesh.
    virtual void compute_vertex_values(std::vector<double>& vertex_values,
                                       const Mesh& mesh) const;

  protected:

    const std::size_t _dim;

    std::vector<Function> _displacements;

  };

}
#endif
