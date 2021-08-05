// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#ifdef HAS_ADIOS2
#include "utils.h"
#include <adios2.h>
#include <string>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace dolfinx
{
namespace mesh
{
class Mesh;
}

namespace fem
{
template <typename T>
class Function;
}

namespace io
{
// Convenience functions for declaring attributes and variables with adios2
namespace adios2_utils
{

//-----------------------------------------------------------------------------
// Extract the mesh connectivity for all cells of a given mesh
xt::xtensor<std::uint64_t, 2>
extract_connectivity(std::shared_ptr<const dolfinx::mesh::Mesh> mesh);

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
//-----------------------------------------------------------------------------
// Write function (real or complex) to to ADIOS2.
// Data is padded to be three dimensional if vector
// and 9 dimensional if tensor
// @param[in] io The ADIOS2 io
// @param[in] engine The ADIOS2 engine
// @param[in] u The function
template <typename Scalar>
void write_function_at_nodes(adios2::IO& io, adios2::Engine& engine,
                             const dolfinx::fem::Function<Scalar>& u)
{

  auto function_data = u.compute_point_values();
  std::uint32_t local_size = function_data.shape(0);
  std::uint32_t block_size = function_data.shape(1);
  // Extract real and imaginary parts
  std::vector<std::string> parts = {""};
  if constexpr (!std::is_scalar<Scalar>::value)
    parts = {"real", "imag"};

  // Write each real and imaginary part of the function
  const int rank = u.function_space()->element()->value_rank();
  const std::uint32_t num_components = std::pow(3, rank);
  std::vector<double> out_data(num_components * local_size);
  for (const auto& part : parts)
  {
    std::string function_name = u.name;
    if (part != "")
      function_name += "_" + part;
    adios2::Variable<double> local_output = DefineVariable<double>(
        io, function_name, {}, {}, {local_size, num_components});

    // Loop over components of each real and imaginary part
    for (size_t i = 0; i < local_size; ++i)
    {
      auto data_row = xt::row(function_data, i);
      for (size_t j = 0; j < block_size; ++j)
      {
        if (part == "imag")
          out_data[i * num_components + j] = std::imag(data_row[j]);
        else
          out_data[i * num_components + j] = std::real(data_row[j]);
      }

      // Pad data to 3D if vector or tensor data
      for (size_t j = block_size; j < num_components; ++j)
        out_data[i * num_components + j] = 0;
    }

    // To reuse out_data, we use sync mode here
    engine.Put<double>(local_output, out_data.data(), adios2::Mode::Sync);
  }
}

} // namespace adios2_utils
} // namespace io
} // namespace dolfinx
#endif