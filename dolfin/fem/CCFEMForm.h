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
// Last changed: 2013-09-19

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

    /// Create empty linear CCFEM variational form (shared pointer version)
    CCFEMForm(boost::shared_ptr<const CCFEMFunctionSpace> function_space);

    /// Create empty linear CCFEM variational form (reference version)
    CCFEMForm(const CCFEMFunctionSpace& function_space);

    /// Create empty bilinear CCFEM variational form (shared pointer version)
    CCFEMForm(boost::shared_ptr<const CCFEMFunctionSpace> function_space_0,
              boost::shared_ptr<const CCFEMFunctionSpace> function_space_1);

    /// Create empty bilinear CCFEM variational form (reference version)
    CCFEMForm(const CCFEMFunctionSpace& function_space_0,
              const CCFEMFunctionSpace& function_space_1);

    /// Destructor
    ~CCFEMForm();

    /// Return rank of form (bilinear form = 2, linear form = 1,
    /// functional = 0, etc)
    ///
    /// *Returns*
    ///     std::size_t
    ///         The rank of the form.
    std::size_t rank() const;

    /// Return the number of forms (parts) of the CCFEM form
    ///
    /// *Returns*
    ///     std::size_t
    ///         The number of forms (parts) of the CCFEM form.
    std::size_t num_parts() const;

    /// Return form (part) number i
    ///
    /// *Returns*
    ///     _Form_
    ///         Form (part) number i.
    boost::shared_ptr<const Form> part(std::size_t i) const;

    /// Return function space for given argument
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         Index
    ///
    /// *Returns*
    ///     _CCFEMFunctionSpace_
    ///         Function space shared pointer.
    boost::shared_ptr<const CCFEMFunctionSpace> function_space(std::size_t i) const;

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

    /// Clear CCFEM form
    void clear();

  private:

    // The rank of the form
    std::size_t _rank;

    // Function spaces (one for each argument)
    std::vector<boost::shared_ptr<const CCFEMFunctionSpace> > _function_spaces;

    // List of forms (one for each part)
    std::vector<boost::shared_ptr<const Form> > _forms;

  };

}

#endif

