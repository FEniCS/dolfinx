// Copyright (C) 2013 Anders Logg
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
// Last changed: 2013-09-12

#ifndef __CCFEM_FORM_H
#define __CCFEM_FORM_H

#include <vector>
#include <boost/shared_ptr.hpp>

namespace dolfin
{

  // Forward declarations
  class CCFEMFunctionSpace;
  class Form;

  /// This class represents a variational form on a cut and composite
  /// finite element function space (CCFEM) defined on one or more
  /// possibly intersecting meshes.
  ///
  /// FIXME: Document usage of class with add() followed by build()

  class CCFEMForm
  {
  public:

    /// Create empty CCFEM variational form (shared pointer version)
    CCFEMForm(boost::shared_ptr<const CCFEMFunctionSpace> function_space);

    /// Create empty CCFEM variational form (reference version)
    CCFEMForm(const CCFEMFunctionSpace& function_space);

    /// Destructor
    ~CCFEMForm();

    /// Add form (shared pointer version)
    ///
    /// *Arguments*
    ///     form (_Form_)
    ///         The form.
    void add(boost::shared_ptr<const Form> form);

    /// Add form (reference version)
    ///
    /// *Arguments*
    ///     form (_Form_)
    ///         The form.
    void add(const Form& form);

    /// Build CCFEM form
    void build();

  private:

    // The function space
    boost::shared_ptr<const CCFEMFunctionSpace> _function_space;

    // List of forms
    std::vector<boost::shared_ptr<const Form> > _forms;

  };

}

#endif

