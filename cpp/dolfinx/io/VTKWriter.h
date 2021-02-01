// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <complex>
#include <string>

namespace dolfinx
{
namespace fem
{
template <typename T>
class Function;
}
namespace mesh
{
class Mesh;
}

namespace io
{

/// Write VTK mesh::Mesh representation

class VTKWriter
{
public:
  /// mesh::Mesh writer
  static void write_mesh(const mesh::Mesh& mesh, std::size_t cell_dim,
                         std::string file);

  /// Cell data writer
  static void write_cell_data(const fem::Function<double>& u, std::string file);

  /// Cell data writer
  static void write_cell_data(const fem::Function<std::complex<double>>& u,
                              std::string file);
};
} // namespace io
} // namespace dolfinx
