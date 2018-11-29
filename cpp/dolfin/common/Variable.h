// Copyright (C) 2003-2009 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>
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

  /// Create variable with given name
  Variable(const std::string name);

  /// Copy constructor
  Variable(const Variable& variable);

  /// Move constructor
  Variable(Variable&& variable) = default;

  /// Destructor
  virtual ~Variable() = default;

  /// Assignment operator
  const Variable& operator=(const Variable& variable);

  /// Rename variable
  void rename(const std::string name);

  /// Return name
  std::string name() const;

  /// Get unique identifier.
  ///
  /// @returns _std::size_t_
  ///         The unique integer identifier associated with the object.
  std::size_t id() const { return unique_id; }

  /// Return informal string representation (pretty-print)
  virtual std::string str(bool verbose) const;

private:
  // Name
  std::string _name;

  // Unique identifier
  const std::size_t unique_id;
};
}
}
