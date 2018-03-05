// Copyright (C) 2003-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>
#include <dolfin/parameter/Parameters.h>
#include <string>

namespace dolfin
{

namespace common
{

/// Common base class for DOLFIN variables.

class Variable
{
public:
  /// Create unnamed variable
  Variable();

  /// Create variable with given name and label
  Variable(const std::string name, const std::string label);

  /// Copy constructor
  Variable(const Variable& variable);

  /// Move constructor
  Variable(Variable&& variable) = default;

  /// Destructor
  virtual ~Variable() = default;

  /// Assignment operator
  const Variable& operator=(const Variable& variable);

  /// Rename variable
  void rename(const std::string name, const std::string label);

  /// Return name
  std::string name() const;

  /// Return label (description)
  std::string label() const;

  /// Get unique identifier.
  ///
  /// *Returns*
  ///     _std::size_t_
  ///         The unique integer identifier associated with the object.
  std::size_t id() const { return unique_id; }

  /// Return informal string representation (pretty-print)
  virtual std::string str(bool verbose) const;

  /// Parameters
  parameter::Parameters parameters;

private:
  // Name
  std::string _name;

  // Label
  std::string _label;

  // Unique identifier
  const std::size_t unique_id;
};
}
}