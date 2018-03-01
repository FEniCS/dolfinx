// Copyright (C) 2013 Johan Hake
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <utility>
#include <vector>

namespace dolfin
{
namespace function
{
class Function;

/// This class represents a linear combination of functions. It is
/// mostly used as an intermediate class for operations such as u =
/// 3*u0 + 4*u1; where the rhs generates an FunctionAXPY.
class FunctionAXPY
{

public:
  /// Enum to decide what way AXPY is constructed
  enum class Direction : int
  {
    ADD_ADD = 0,
    SUB_ADD = 1,
    ADD_SUB = 2,
    SUB_SUB = 3
  };

  /// Constructor
  FunctionAXPY(std::shared_ptr<const Function> func, double scalar);

  /// Constructor
  FunctionAXPY(const FunctionAXPY& axpy, double scalar);

  /// Constructor
  FunctionAXPY(std::shared_ptr<const Function> func0,
               std::shared_ptr<const Function> func1, Direction direction);

  /// Constructor
  FunctionAXPY(const FunctionAXPY& axpy, std::shared_ptr<const Function> func,
               Direction direction);

  /// Constructor
  FunctionAXPY(const FunctionAXPY& axpy0, const FunctionAXPY& axpy1,
               Direction direction);

  /// Constructor
  FunctionAXPY(
      std::vector<std::pair<double, std::shared_ptr<const Function>>> pairs);

  /// Copy constructor
  FunctionAXPY(const FunctionAXPY& axpy);

  /// Destructor
  ~FunctionAXPY();

  /// Addition operator
  FunctionAXPY operator+(std::shared_ptr<const Function> func) const;

  /// Addition operator
  FunctionAXPY operator+(const FunctionAXPY& axpy) const;

  /// Subtraction operator
  FunctionAXPY operator-(std::shared_ptr<const Function> func) const;

  /// Subtraction operator
  FunctionAXPY operator-(const FunctionAXPY& axpy) const;

  /// Scale operator
  FunctionAXPY operator*(double scale) const;

  /// Scale operator
  FunctionAXPY operator/(double scale) const;

  /// Return the scalar and Function pairs
  const std::vector<std::pair<double, std::shared_ptr<const Function>>>&
  pairs() const;

private:
  /// Register another AXPY object
  void _register(const FunctionAXPY& axpy0, double scale);

  std::vector<std::pair<double, std::shared_ptr<const Function>>> _pairs;
};
}
}