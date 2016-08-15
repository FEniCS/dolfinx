// Copyright (C) 2013-2016 Anders Logg
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
// First added:  2013-09-12
// Last changed: 2016-03-02

#ifndef __MULTI_MESH_FORM_H
#define __MULTI_MESH_FORM_H

#include <vector>
#include <memory>

namespace dolfin
{

  // Forward declarations
  class MultiMeshFunctionSpace;
  class MultiMesh;
  class Form;

  /// This class represents a variational form on a cut and composite
  /// finite element function space (MultiMesh) defined on one or more
  /// possibly intersecting meshes.

  class MultiMeshForm
  {
  public:

    // FIXME: Test multimesh functionals. Should likely require a multimesh
    // when instaniated and this constructor should then be removed.
    MultiMeshForm() {}

    /// Create empty multimesh functional
    MultiMeshForm(std::shared_ptr<const MultiMesh> multimesh);

    /// Create empty linear multimesh variational form
    MultiMeshForm(std::shared_ptr<const MultiMeshFunctionSpace> function_space);

    /// Create empty bilinear multimesh variational form
    MultiMeshForm(std::shared_ptr<const MultiMeshFunctionSpace> function_space_0,
                  std::shared_ptr<const MultiMeshFunctionSpace> function_space_1);

    /// Destructor
    ~MultiMeshForm();

    /// Return rank of form (bilinear form = 2, linear form = 1,
    /// functional = 0, etc)
    ///
    /// *Returns*
    ///     std::size_t
    ///         The rank of the form.
    std::size_t rank() const;

    /// Return the number of forms (parts) of the MultiMesh form
    ///
    /// *Returns*
    ///     std::size_t
    ///         The number of forms (parts) of the MultiMesh form.
    std::size_t num_parts() const;

    /// Extract common multimesh from form
    ///
    /// *Returns*
    ///     _MultiMesh_
    ///         The mesh.
    std::shared_ptr<const MultiMesh> multimesh() const;

    /// Return form (part) number i
    ///
    /// *Returns*
    ///     _Form_
    ///         Form (part) number i.
    std::shared_ptr<const Form> part(std::size_t i) const;

    /// Return function space for given argument
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         Index
    ///
    /// *Returns*
    ///     _MultiMeshFunctionSpace_
    ///         Function space shared pointer.
    std::shared_ptr<const MultiMeshFunctionSpace> function_space(std::size_t i) const;

    /// Add form (shared pointer version)
    ///
    /// *Arguments*
    ///     form (_Form_)
    ///         The form.
    void add(std::shared_ptr<const Form> form);

    /// Build MultiMesh form
    void build();

    /// Clear MultiMesh form
    void clear();

  private:

    // The rank of the form
    std::size_t _rank;

    // Multimesh
    std::shared_ptr<const MultiMesh> _multimesh;

    // Function spaces (one for each argument)
    std::vector<std::shared_ptr<const MultiMeshFunctionSpace>> _function_spaces;

    // List of forms (one for each part)
    std::vector<std::shared_ptr<const Form>> _forms;

  };

}

#endif
