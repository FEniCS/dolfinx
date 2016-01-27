// Copyright (C) 2008-2009 Anders Logg
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

#ifndef __COEFFICIENT_ASSIGNER_H
#define __COEFFICIENT_ASSIGNER_H

#include <cstddef>
#include <memory>

namespace dolfin
{

  class Form;
  class GenericFunction;

  /// This class is used for assignment of coefficients to
  /// forms, which allows magic like
  ///
  ///   a.f = f
  ///   a.g = g
  ///
  /// which will insert the coefficients f and g in the correct
  /// positions in the list of coefficients for the form.

  class CoefficientAssigner
  {
  public:

    /// Create coefficient assigner for coefficient with given number
    CoefficientAssigner(Form& form, std::size_t number);

    /// Destructor
    ~CoefficientAssigner();

    /// Assign coefficient
    void operator= (std::shared_ptr<const GenericFunction> coefficient);

  private:

    // The form
    Form& _form;

    // The number of the coefficient
    std::size_t _number;

  };

}

#endif
