// Copyright (C) 2015 Anders Logg
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
// First added:  2015-11-05
// Last changed: 2015-11-05

#ifndef __MULTIMESH_COEFFICIENT_ASSIGNER_H
#define __MULTIMESH_COEFFICIENT_ASSIGNER_H

#include <cstddef>

namespace dolfin
{

  class MultiMeshForm;
  class GenericFunction;

  /// This class is used for assignment of multimesh coefficients to
  /// forms, which allows magic like
  ///
  ///   a.f = f
  ///   a.g = g
  ///
  /// which will insert the coefficients f and g in the correct
  /// positions in the list of coefficients for the form.
  ///
  /// Note that coefficients can also be assigned manually to the
  /// individual parts of a multimesh form. Assigning to a multimesh
  /// coefficient assigner will assign the same coefficient to all
  /// parts of a form.

  class MultiMeshCoefficientAssigner
  {
  public:

    /// Create multimesh coefficient assigner for coefficient with given number
    MultiMeshCoefficientAssigner(MultiMeshForm& form, std::size_t number);

    /// Destructor
    ~MultiMeshCoefficientAssigner();

    /// Assign coefficient
    void operator= (const GenericFunction& coefficient);

  private:

    // The multimesh form
    MultiMeshForm& _form;

    // The number of the coefficient
    std::size_t _number;

  };

}

#endif
