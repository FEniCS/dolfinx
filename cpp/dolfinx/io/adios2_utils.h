// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#include <string>
#include <vector>

namespace adios2
{
class IO;
template <class T>
class Attribute;
using Dims = std::vector<size_t>;
template <class T>
class Variable;

enum class Mode;
} // namespace adios2

namespace dolfinx
{
namespace io
{
enum class mode;
// Convenience functions for declaring attributes and variables with adios2
namespace adios2_utils
{

// Safe definition of an attsibute. First check if it has already been defined
// and return it. If not defined create new attribute.
template <class T>
adios2::Attribute<T> DefineAttribute(adios2::IO& io, const std::string& name,
                                     const T& value,
                                     const std::string& var_name = "",
                                     const std::string& separator = "/");

// Safe definition of a variable. First check if it has already been defined
// and return it. If not defined create new variable.
template <class T>
adios2::Variable<T> DefineVariable(adios2::IO& io, const std::string& name,
                                   const adios2::Dims& shape = adios2::Dims(),
                                   const adios2::Dims& start = adios2::Dims(),
                                   const adios2::Dims& count = adios2::Dims());

// Convert DOLFINx io mode to ADIOS2 mode
adios2::Mode dolfinx_to_adios_mode(mode mode);

} // namespace adios2_utils
} // namespace io
} // namespace dolfinx