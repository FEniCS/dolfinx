// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#include <string>
#include <vector>

#ifdef HAS_ADIOS2
#include <adios2.h>

namespace dolfinx
{
namespace io
{
enum class mode;
// Convenience functions for declaring attributes and variables with adios2
namespace adios2_utils
{

//-----------------------------------------------------------------------------
// Safe definition of an attribute. First check if it has already been defined
// and return it. If not defined create new attribute.
template <class T>
adios2::Attribute<T> DefineAttribute(adios2::IO& io, const std::string& name,
                                     const T& value,
                                     const std::string& var_name = "",
                                     const std::string& separator = "/")
{
  if (adios2::Attribute<T> attr = io.InquireAttribute<T>(name); attr)
    return attr;
  else
    return io.DefineAttribute<T>(name, value, var_name, separator);
}

//-----------------------------------------------------------------------------
// Safe definition of a variable. First check if it has already been defined
// and return it. If not defined create new variable.
template <class T>
adios2::Variable<T> DefineVariable(adios2::IO& io, const std::string& name,
                                   const adios2::Dims& shape = adios2::Dims(),
                                   const adios2::Dims& start = adios2::Dims(),
                                   const adios2::Dims& count = adios2::Dims())
{
  adios2::Variable<T> v = io.InquireVariable<T>(name);
  if (v)
  {
    if (v.Count() != count and v.ShapeID() == adios2::ShapeID::LocalArray)
      v.SetSelection({start, count});
  }
  else
    v = io.DefineVariable<T>(name, shape, start, count);

  return v;
}

// Convert DOLFINx io mode to ADIOS2 mode
constexpr adios2::Mode dolfinx_to_adios_mode(mode mode)
{
  switch (mode)
  {
  case mode::write:
    return adios2::Mode::Write;
  case mode::append:
    return adios2::Mode::Append;
  case mode::read:
    throw std::runtime_error("Unsupported file mode");
  //   return adios2::Mode::Read;
  default:
    throw std::runtime_error("Unknown file mode");
  }
}

} // namespace adios2_utils
} // namespace io
} // namespace dolfinx
#endif