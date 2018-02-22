// Copyright (C) 2008-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

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
  void operator=(std::shared_ptr<const GenericFunction> coefficient);

private:
  // The form
  Form& _form;

  // The number of the coefficient
  std::size_t _number;
};
}
