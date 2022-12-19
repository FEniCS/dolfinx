// Copyright (C) 2005-2020 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKFile.h"
#include "cells.h"
#include "vtk_utils.h"
#include "xdmf_utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <filesystem>
#include <iterator>
#include <pugixml.hpp>
#include <span>
#include <sstream>
#include <string>

using namespace dolfinx;

namespace
{
/// String suffix for real and complex components of a vector-valued
/// field
constexpr std::array field_ext = {"_real", "_imag"};

//----------------------------------------------------------------------------
/// Return true if Function is a cell-wise constant, otherwise false
bool is_cellwise(const fem::FunctionSpace& V)
{
  assert(V.element());
  const int rank = V.element()->value_shape().size();
  assert(V.mesh());
  const int tdim = V.mesh()->topology().dim();

  // cell_based_dim = tdim^rank
  int cell_based_dim = 1;
  for (int i = 0; i < rank; ++i)
    cell_based_dim *= tdim;

  assert(V.dofmap());
  if (V.dofmap()->element_dof_layout().num_dofs() == cell_based_dim)
    return true;
  else
    return false;
}
//----------------------------------------------------------------------------

/// Get counter string to include in filename
std::string get_counter(const pugi::xml_node& node, const std::string& name)
{
  // Count number of entries
  const size_t n = std::distance(node.children(name.c_str()).begin(),
                                 node.children(name.c_str()).end());

  // Compute counter string
  constexpr int num_digits = 6;
  std::string counter = std::to_string(n);
  return std::string(num_digits - counter.size(), '0').append(counter);
}
//----------------------------------------------------------------------------

/// Convert a container to a std::stringstream
template <typename T>
std::stringstream container_to_string(const T& x, int precision)
{
  std::stringstream s;
  s.precision(precision);
  std::for_each(x.begin(), x.end(), [&s](auto e) { s << e << " "; });
  return s;
}
//----------------------------------------------------------------------------

void add_pvtu_mesh(pugi::xml_node& node)
{
  // -- Cell data (PCellData)
  pugi::xml_node cell_data_node = node.child("PCellData");
  if (cell_data_node.empty())
    cell_data_node = node.append_child("PCellData");

  pugi::xml_node cell_array_node = cell_data_node.append_child("PDataArray");
  cell_array_node.append_attribute("type") = "UInt8";
  cell_array_node.append_attribute("Name") = "vtkGhostType";

  pugi::xml_node cell_id_node = cell_data_node.append_child("PDataArray");
  cell_id_node.append_attribute("type") = "Int64";
  cell_id_node.append_attribute("Name") = "vtkOriginalCellIds";
  cell_id_node.append_attribute("IdType") = "1";

  // -- Point data (PPointData)
  pugi::xml_node point_data_node = node.child("PPointData");
  if (point_data_node.empty())
    point_data_node = node.append_child("PPointData");

  pugi::xml_node point_id_node = point_data_node.append_child("PDataArray");
  point_id_node.append_attribute("type") = "Int64";
  point_id_node.append_attribute("Name") = "vtkOriginalPointIds";
  point_id_node.append_attribute("IdType") = "1";

  // Ghost points
  pugi::xml_node point_ghost_node = point_data_node.append_child("PDataArray");
  point_ghost_node.append_attribute("type") = "UInt8";
  point_ghost_node.append_attribute("Name") = "vtkGhostType";

  // -- Points (PPoints)
  pugi::xml_node x_data_node = node.child("PPoints");
  if (x_data_node.empty())
    x_data_node = node.append_child("PPoints");
  pugi::xml_node data_node = x_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Float64";
  data_node.append_attribute("NumberOfComponents") = "3";
}
//----------------------------------------------------------------------------

/// Add float data to a pugixml node
/// @param[in] name The name of the data array
/// @param[in] rank The rank of the field, e.g. 1 for a vector and 2 for
/// a tensor
/// @param[in] values The data array to add
/// @param[in,out] data_node The XML node to add data to
template <typename T>
void add_data_float(const std::string& name, int rank,
                    std::span<const T> values, pugi::xml_node& node)
{
  static_assert(std::is_floating_point_v<T>, "Scalar must be a float");

  constexpr int size = 8 * sizeof(T);
  std::string type = std::string("Float") + std::to_string(size);

  pugi::xml_node field_node = node.append_child("DataArray");
  field_node.append_attribute("type") = type.c_str();
  field_node.append_attribute("Name") = name.c_str();
  field_node.append_attribute("format") = "ascii";

  if (rank == 1)
    field_node.append_attribute("NumberOfComponents") = 3;
  else if (rank == 2)
    field_node.append_attribute("NumberOfComponents") = 9;
  field_node.append_child(pugi::node_pcdata)
      .set_value(container_to_string(values, 16).str().c_str());
}
//----------------------------------------------------------------------------

/// At data to a pugixml node
/// @note If `values` is complex, two data arrays will be added (one
/// real and one complex), with suffixes from `field_ext` added to the
/// name
/// @param[in] name The name of the data array
/// @param[in] rank The rank of the field, e.g. 1 for a vector and 2 for
/// a tensor
/// @param[in] values The data array to add
/// @param[in,out] data_node The XML node to add data to
template <typename T>
void add_data(const std::string& name, int rank, std::span<const T> values,
              pugi::xml_node& node)
{
  if constexpr (std::is_scalar_v<T>)
    add_data_float(name, rank, values, node);
  else
  {
    using U = typename T::value_type;
    std::vector<U> v(values.size());

    std::transform(values.begin(), values.end(), v.begin(),
                   [](auto x) { return x.real(); });
    add_data_float(name + field_ext[0], rank, std::span<const U>(v), node);

    std::transform(values.begin(), values.end(), v.begin(),
                   [](auto x) { return x.imag(); });
    add_data_float(name + field_ext[1], rank, std::span<const U>(v), node);
  }
}
//----------------------------------------------------------------------------

/// Add mesh geometry and topology data to a pugixml node. This function
/// adds the Points and Cells nodes to the input node.
/// @param[in] x Coordinates of the points, row-major storage
/// @param[in] xshape The shape of `x`
/// @param[in] x_id Unique global index for each point
/// @param[in] x_ghost Flag indicating if a point is a owned (0) or is a
/// ghost (1)
/// @param[in] cells The mesh topology
/// @param[in] cellmap The index map for the cells
/// @param[in] celltype The cell type
/// @param[in] tdim Topological dimension of the cells
/// @param[in,out] piece_node The XML node to add data to
void add_mesh(std::span<const double> x, std::array<std::size_t, 2> /*xshape*/,
              std::span<const std::int64_t> x_id,
              std::span<const std::uint8_t> x_ghost,
              std::span<const std::int64_t> cells,
              std::array<std::size_t, 2> cshape,
              const common::IndexMap& cellmap, mesh::CellType celltype,
              int tdim, pugi::xml_node& piece_node)
{
  // -- Add geometry (points)

  pugi::xml_node points_node = piece_node.append_child("Points");
  pugi::xml_node x_node = points_node.append_child("DataArray");
  x_node.append_attribute("type") = "Float64";
  x_node.append_attribute("NumberOfComponents") = "3";
  x_node.append_attribute("format") = "ascii";
  x_node.append_child(pugi::node_pcdata)
      .set_value(container_to_string(x, 16).str().c_str());

  // -- Add topology (cells)

  pugi::xml_node cells_node = piece_node.append_child("Cells");
  pugi::xml_node connectivity_node = cells_node.append_child("DataArray");
  connectivity_node.append_attribute("type") = "Int32";
  connectivity_node.append_attribute("Name") = "connectivity";
  connectivity_node.append_attribute("format") = "ascii";
  {
    std::stringstream ss;
    std::for_each(cells.begin(), cells.end(),
                  [&ss](auto& v) { ss << v << " "; });
    connectivity_node.append_child(pugi::node_pcdata)
        .set_value(ss.str().c_str());
  }

  pugi::xml_node offsets_node = cells_node.append_child("DataArray");
  offsets_node.append_attribute("type") = "Int32";
  offsets_node.append_attribute("Name") = "offsets";
  offsets_node.append_attribute("format") = "ascii";
  {
    std::stringstream ss;
    int num_nodes = cshape[1];
    for (std::size_t i = 0; i < cshape[0]; ++i)
      ss << (i + 1) * num_nodes << " ";
    offsets_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
  }

  pugi::xml_node type_node = cells_node.append_child("DataArray");
  type_node.append_attribute("type") = "Int8";
  type_node.append_attribute("Name") = "types";
  type_node.append_attribute("format") = "ascii";
  int vtk_celltype = io::cells::get_vtk_cell_type(celltype, tdim);
  {
    std::stringstream ss;
    for (std::size_t c = 0; c < cshape[0]; ++c)
      ss << vtk_celltype << " ";
    type_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
  }

  // Ghost cell markers
  pugi::xml_node cells_data_node = piece_node.append_child("CellData");

  pugi::xml_node ghost_cell_node = cells_data_node.append_child("DataArray");
  ghost_cell_node.append_attribute("type") = "UInt8";
  ghost_cell_node.append_attribute("Name") = "vtkGhostType";
  ghost_cell_node.append_attribute("format") = "ascii";
  ghost_cell_node.append_attribute("RangeMin") = "0";
  ghost_cell_node.append_attribute("RangeMax") = "1";
  {
    std::stringstream ss;
    for (std::int32_t c = 0; c < cellmap.size_local(); ++c)
      ss << 0 << " ";
    for (std::size_t c = cellmap.size_local(); c < cshape[0]; ++c)
      ss << 1 << " ";
    ghost_cell_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
  }

  // Original cell IDs
  pugi::xml_node cell_id_node = cells_data_node.append_child("DataArray");
  cell_id_node.append_attribute("type") = "Int64";
  cell_id_node.append_attribute("IdType") = "1";
  cell_id_node.append_attribute("Name") = "vtkOriginalCellIds";
  cell_id_node.append_attribute("format") = "ascii";
  {
    std::stringstream ss;
    const std::int64_t cell_offset = cellmap.local_range()[0];
    for (std::int32_t c = 0; c < cellmap.size_local(); ++c)
      ss << cell_offset + c << " ";
    std::for_each(cellmap.ghosts().begin(), cellmap.ghosts().end(),
                  [&ss](auto& idx) { ss << idx << " "; });
    cell_id_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
  }

  auto [min_idx, max_idx] = cellmap.local_range();
  max_idx -= 1;
  if (!cellmap.ghosts().empty())
  {
    auto& ghosts = cellmap.ghosts();
    auto minmax = std::minmax_element(ghosts.begin(), ghosts.end());
    min_idx = std::min(min_idx, *minmax.first);
    max_idx = std::max(max_idx, *minmax.second);
  }
  cell_id_node.append_attribute("RangeMin") = min_idx;
  cell_id_node.append_attribute("RangeMax") = max_idx;

  pugi::xml_node points_data_node = piece_node.append_child("PointData");

  // Original point IDs
  pugi::xml_node point_id_node = points_data_node.append_child("DataArray");
  point_id_node.append_attribute("type") = "Int64";
  point_id_node.append_attribute("IdType") = "1";
  point_id_node.append_attribute("Name") = "vtkOriginalPointIds";
  point_id_node.append_attribute("format") = "ascii";
  {
    std::stringstream ss;
    std::for_each(x_id.begin(), x_id.end(),
                  [&ss](auto idx) { ss << idx << " "; });
    point_id_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
  }
  if (!x_id.empty())
  {
    auto minmax = std::minmax_element(x_id.begin(), x_id.end());
    point_id_node.append_attribute("RangeMin") = *minmax.first;
    point_id_node.append_attribute("RangeMax") = *minmax.second;
  }

  // Point ghosts
  pugi::xml_node point_ghost_node = points_data_node.append_child("DataArray");
  point_ghost_node.append_attribute("type") = "UInt8";
  point_ghost_node.append_attribute("Name") = "vtkGhostType";
  point_ghost_node.append_attribute("format") = "ascii";
  {
    std::stringstream ss;
    std::for_each(x_ghost.begin(), x_ghost.end(),
                  [&ss](int ghost) { ss << ghost << " "; });
    point_ghost_node.append_child(pugi::node_pcdata)
        .set_value(ss.str().c_str());
  }
  if (!x_ghost.empty())
  {
    auto minmax = std::minmax_element(x_ghost.begin(), x_ghost.end());
    point_ghost_node.append_attribute("RangeMin") = *minmax.first;
    point_ghost_node.append_attribute("RangeMax") = *minmax.second;
  }
}
//----------------------------------------------------------------------------
template <typename T>
void write_function(
    const std::vector<std::reference_wrapper<const fem::Function<T>>>& u,
    double time, std::unique_ptr<pugi::xml_document>& xml_doc,
    const std::filesystem::path& filename)
{
  if (!xml_doc)
    throw std::runtime_error("VTKFile has been closed");
  if (u.empty())
    return;

  // Extract the first function space with pointwise data. If no
  // pointwise functions, take first FunctionSpace
  auto V0 = u.front().get().function_space();
  assert(V0);
  for (auto& v : u)
  {
    auto V = v.get().function_space();
    assert(V);
    if (!is_cellwise(*V))
    {
      V0 = V;
      break;
    }
  }

  // Check compatibility for all functions
  auto mesh0 = V0->mesh();
  assert(mesh0);
  auto element0 = V0->element();
  for (auto& v : u)
  {
    auto V = v.get().function_space();
    assert(V);

    // Check that functions share common mesh
    assert(V->mesh());
    if (V->mesh() != mesh0)
    {
      throw std::runtime_error(
          "All Functions written to VTK file must share the same Mesh.");
    }

    // Check that v isn't a sub-function
    if (!V->component().empty())
      throw std::runtime_error("Cannot write sub-Functions to VTK file.");

    // Check that pointwise elements are the same (up to the block size)
    if (!is_cellwise(*V))
    {
      if (*(V->element()) != *element0)
      {
        throw std::runtime_error("All point-wise Functions written to VTK file "
                                 "must have same element.");
      }
    }
  }

  // Get the PVD "Collection" node
  pugi::xml_node xml_collections
      = xml_doc->child("VTKFile").child("Collection");
  assert(xml_collections);

  // Compute counter string
  const std::string counter_str = get_counter(xml_collections, "DataSet");

  // Create a VTU XML object
  pugi::xml_document xml_vtu;
  pugi::xml_node vtk_node_vtu = xml_vtu.append_child("VTKFile");
  vtk_node_vtu.append_attribute("type") = "UnstructuredGrid";
  vtk_node_vtu.append_attribute("version") = "2.2";
  pugi::xml_node grid_node_vtu = vtk_node_vtu.append_child("UnstructuredGrid");

  // Build mesh data using first FunctionSpace
  std::vector<double> x;
  std::array<std::size_t, 2> xshape;
  std::vector<std::int64_t> x_id;
  std::vector<std::uint8_t> x_ghost;
  std::vector<std::int64_t> cells;
  std::array<std::size_t, 2> cshape;
  if (is_cellwise(*V0))
  {
    std::vector<std::int64_t> tmp;
    std::tie(tmp, cshape) = io::extract_vtk_connectivity(*mesh0);
    cells.assign(tmp.begin(), tmp.end());
    const mesh::Geometry& geometry = mesh0->geometry();
    x.assign(geometry.x().begin(), geometry.x().end());
    xshape = {geometry.x().size() / 3, 3};
    x_id = geometry.input_global_indices();
    auto xmap = geometry.index_map();
    assert(xmap);
    x_ghost.resize(xshape[0], 0);
    std::fill(std::next(x_ghost.begin(), xmap->size_local()), x_ghost.end(), 1);
  }
  else
    std::tie(x, xshape, x_id, x_ghost, cells, cshape)
        = io::vtk_mesh_from_space(*V0);

  // Add "Piece" node and required metadata
  pugi::xml_node piece_node = grid_node_vtu.append_child("Piece");
  piece_node.append_attribute("NumberOfPoints") = xshape[0];
  piece_node.append_attribute("NumberOfCells") = cshape[0];

  // Add mesh data to "Piece" node
  int tdim = mesh0->topology().dim();
  add_mesh(x, xshape, x_id, x_ghost, cells, cshape,
           *mesh0->topology().index_map(tdim), mesh0->topology().cell_type(),
           mesh0->topology().dim(), piece_node);

  // FIXME: is this actually setting the first?
  // Set last scalar/vector/tensor Functions in u to be the 'active'
  // (default) field(s)
  constexpr std::array tensor_str = {"Scalars", "Vectors", "Tensors"};
  for (auto _u : u)
  {
    auto V = _u.get().function_space();
    assert(V);
    auto data_type = is_cellwise(*V) ? "CellData" : "PointData";
    if (piece_node.child(data_type).empty())
      piece_node.append_child(data_type);

    const int rank = V->element()->value_shape().size();
    pugi::xml_node data_node = piece_node.child(data_type);
    if (data_node.attribute(tensor_str[rank]).empty())
      data_node.append_attribute(tensor_str[rank]);
    pugi::xml_attribute data = data_node.attribute(tensor_str[rank]);
    data.set_value(_u.get().name.c_str());
  }

  // Add cell/point data to VTU node
  for (auto _u : u)
  {
    auto V = _u.get().function_space();
    auto element = V->element();
    int rank = element->value_shape().size();
    std::int32_t num_comp = std::pow(3, rank);
    if (is_cellwise(*V))
    {
      // -- Cell-wise data

      pugi::xml_node data_node = piece_node.child("CellData");
      assert(!data_node.empty());
      auto dofmap = V->dofmap();
      int bs = dofmap->bs();
      std::vector<T> data(cshape[0] * num_comp, 0);
      auto u_vector = _u.get().x()->array();
      for (std::size_t c = 0; c < cshape[0]; ++c)
      {
        std::span<const std::int32_t> dofs = dofmap->cell_dofs(c);
        for (std::size_t i = 0; i < dofs.size(); ++i)
          for (int k = 0; k < bs; ++k)
            data[num_comp * c + k] = u_vector[bs * dofs[i] + k];
      }

      add_data(_u.get().name, rank, std::span<const T>(data), data_node);
    }
    else
    {
      // -- Point-wise data

      pugi::xml_node data_node = piece_node.child("PointData");
      assert(!data_node.empty());

      // Function to pack data to 3D with 'zero' padding, typically when
      // a Function is 2D
      auto pad_data
          = [num_comp](const fem::FunctionSpace& V, std::span<const T> u)
      {
        auto dofmap = V.dofmap();
        int bs = dofmap->bs();
        auto map = dofmap->index_map;
        int map_bs = dofmap->index_map_bs();
        std::int32_t num_dofs_block
            = map_bs * (map->size_local() + map->num_ghosts()) / bs;
        std::vector<T> data(num_dofs_block * num_comp, 0);
        for (int i = 0; i < num_dofs_block; ++i)
        {
          std::copy_n(std::next(u.begin(), i * map_bs), map_bs,
                      std::next(data.begin(), i * num_comp));
        }

        return data;
      };

      if (V == V0)
      {
        // -- Identical spaces
        if (mesh0->geometry().dim() == 3)
          add_data(_u.get().name, rank, _u.get().x()->array(), data_node);
        else
        {
          // Pad with zeros and then add
          auto data = pad_data(*V, _u.get().x()->array());
          add_data(_u.get().name, rank, std::span<const T>(data), data_node);
        }
      }
      else if (*element == *element0)
      {
        // -- Same element, possibly different dofmaps

        // Get dofmaps
        auto dofmap0 = V0->dofmap();
        assert(dofmap0);
        auto dofmap = V->dofmap();
        assert(dofmap);
        int bs = dofmap->bs();

        // Interpolate on each cell
        auto u_vector = _u.get().x()->array();
        std::vector<T> u(u_vector.size());
        for (std::size_t c = 0; c < cshape[0]; ++c)
        {
          std::span<const std::int32_t> dofs0 = dofmap0->cell_dofs(c);
          std::span<const std::int32_t> dofs = dofmap->cell_dofs(c);
          for (std::size_t i = 0; i < dofs0.size(); ++i)
          {
            for (int k = 0; k < bs; ++k)
            {
              assert(i < dofs0.size());
              assert(bs * dofs0[i] + k < (int)u.size());
              u[bs * dofs0[i] + k] = u_vector[bs * dofs[i] + k];
            }
          }
        }

        // Pack/add data
        if (mesh0->geometry().dim() == 3)
          add_data(_u.get().name, rank, std::span<const T>(u), data_node);
        else
        {
          // Pad with zeros and then add
          auto data = pad_data(*V, _u.get().x()->array());
          add_data(_u.get().name, rank, std::span<const T>(data), data_node);
        }
      }
      else
      {
        throw std::runtime_error(
            "Elements differ, not permitted for VTK output");
      }
    }
  }

  // Create filepath for a .vtu file
  auto create_vtu_path = [file_root = filename.parent_path(),
                          file_name = filename.stem(), counter_str](int rank)
  {
    std::filesystem::path vtu = file_root / file_name;
    vtu += +"_p" + std::to_string(rank) + "_" + counter_str;
    vtu.replace_extension("vtu");
    return vtu;
  };

  // Save VTU XML to file
  const int mpi_rank = dolfinx::MPI::rank(mesh0->comm());
  std::filesystem::path vtu = create_vtu_path(mpi_rank);
  if (vtu.has_parent_path())
    std::filesystem::create_directories(vtu.parent_path());
  xml_vtu.save_file(vtu.c_str(), "  ");

  // -- Create a PVTU XML object on rank 0
  std::filesystem::path p_pvtu = filename.parent_path() / filename.stem();
  if (mpi_rank == 0)
  {
    p_pvtu += counter_str;
    p_pvtu.replace_extension("pvtu");

    pugi::xml_document xml_pvtu;
    pugi::xml_node vtk_node = xml_pvtu.append_child("VTKFile");
    vtk_node.append_attribute("type") = "PUnstructuredGrid";
    vtk_node.append_attribute("version") = "1.0";
    pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
    grid_node.append_attribute("GhostLevel") = 1;
    for (auto _u : u)
    {
      if (is_cellwise(*_u.get().function_space()))
      {
        if (grid_node.child("PCellData").empty())
          grid_node.append_child("PCellData");
      }
      else
      {
        if (grid_node.child("PPointData").empty())
          grid_node.append_child("PPointData");
      }
    }

    // Add mesh metadata to PVTU object
    add_pvtu_mesh(grid_node);

    const int mpi_size = dolfinx::MPI::size(mesh0->comm());
    for (auto _u : u)
    {
      auto V = _u.get().function_space();
      std::string d_type = is_cellwise(*V) ? "PCellData" : "PPointData";
      pugi::xml_node data_pnode = grid_node.child(d_type.c_str());
      const int rank = V->element()->value_shape().size();
      constexpr std::array ncomps = {0, 3, 9};

      auto add_field = [&](const std::string& name, int size)
      {
        std::string type = std::string("Float") + std::to_string(size);
        pugi::xml_node data_node = data_pnode.append_child("PDataArray");
        data_node.append_attribute("type") = type.c_str();
        data_node.append_attribute("Name") = name.c_str();
        data_node.append_attribute("NumberOfComponents") = ncomps[rank];
      };

      if constexpr (std::is_scalar_v<T>)
      {
        constexpr int size = 8 * sizeof(T);
        add_field(_u.get().name, size);
      }
      else
      {
        constexpr int size = 8 * sizeof(typename T::value_type);
        add_field(_u.get().name + field_ext[0], size);
        add_field(_u.get().name + field_ext[1], size);
      }

      // Add data for each process to the PVTU object
      for (int r = 0; r < mpi_size; ++r)
      {
        std::filesystem::path vtu = create_vtu_path(r);
        pugi::xml_node piece_node = grid_node.append_child("Piece");
        piece_node.append_attribute("Source") = vtu.filename().c_str();
      }
    }

    // Write PVTU file
    if (p_pvtu.has_parent_path())
      std::filesystem::create_directories(p_pvtu.parent_path());
    xml_pvtu.save_file(p_pvtu.c_str(), "  ");
  }

  // Append PVD file
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file") = p_pvtu.filename().c_str();
}
//----------------------------------------------------------------------------

} // namespace

//----------------------------------------------------------------------------
io::VTKFile::VTKFile(MPI_Comm comm, const std::filesystem::path& filename,
                     const std::string&)
    : _filename(filename), _comm(comm)
{
  _pvd_xml = std::make_unique<pugi::xml_document>();
  assert(_pvd_xml);
  pugi::xml_node vtk_node = _pvd_xml->append_child("VTKFile");
  vtk_node.append_attribute("type") = "Collection";
  vtk_node.append_attribute("version") = "1.0";
  vtk_node.append_child("Collection");
}
//----------------------------------------------------------------------------
io::VTKFile::~VTKFile()
{
  if (_pvd_xml and dolfinx::MPI::rank(_comm.comm()) == 0)
  {
    if (_filename.has_parent_path())
      std::filesystem::create_directories(_filename.parent_path());
    _pvd_xml->save_file(_filename.c_str(), "  ");
  }
}
//----------------------------------------------------------------------------
void io::VTKFile::close()
{
  if (_pvd_xml and dolfinx::MPI::rank(_comm.comm()) == 0)
  {
    if (_filename.has_parent_path())
      std::filesystem::create_directories(_filename.parent_path());

    bool status = _pvd_xml->save_file(_filename.c_str(), "  ");
    if (status == false)
    {
      throw std::runtime_error(
          "Could not write VTKFile. Does the directory "
          "exists and do you have read/write permissions?");
    }
  }
}
//----------------------------------------------------------------------------
void io::VTKFile::flush()
{
  if (!_pvd_xml and dolfinx::MPI::rank(_comm.comm()) == 0)
    throw std::runtime_error("VTKFile has already been closed");

  if (MPI::rank(_comm.comm()) == 0)
  {
    if (_filename.has_parent_path())
      std::filesystem::create_directories(_filename.parent_path());
    _pvd_xml->save_file(_filename.c_str(), "  ");
  }
}
//----------------------------------------------------------------------------
void io::VTKFile::write(const mesh::Mesh& mesh, double time)
{
  if (!_pvd_xml)
    throw std::runtime_error("VTKFile has already been closed");

  // Get the PVD "Collection" node
  pugi::xml_node xml_collections
      = _pvd_xml->child("VTKFile").child("Collection");
  assert(xml_collections);

  // Compute counter string
  const std::string counter_str = get_counter(xml_collections, "DataSet");

  // Get mesh data for this rank
  const mesh::Topology& topology = mesh.topology();
  const mesh::Geometry& geometry = mesh.geometry();
  auto xmap = geometry.index_map();
  assert(xmap);
  const int tdim = topology.dim();
  const std::int32_t num_points = xmap->size_local() + xmap->num_ghosts();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local()
                                 + topology.index_map(tdim)->num_ghosts();

  // Create a VTU XML object
  pugi::xml_document xml_vtu;
  pugi::xml_node vtk_node_vtu = xml_vtu.append_child("VTKFile");
  vtk_node_vtu.append_attribute("type") = "UnstructuredGrid";
  vtk_node_vtu.append_attribute("version") = "2.2";
  pugi::xml_node grid_node_vtu = vtk_node_vtu.append_child("UnstructuredGrid");

  // Add "Piece" node and required metadata
  pugi::xml_node piece_node = grid_node_vtu.append_child("Piece");
  piece_node.append_attribute("NumberOfPoints") = num_points;
  piece_node.append_attribute("NumberOfCells") = num_cells;

  // Add mesh data to "Piece" node
  const auto [cells, cshape] = extract_vtk_connectivity(mesh);
  std::array<std::size_t, 2> xshape = {geometry.x().size() / 3, 3};
  std::vector<std::uint8_t> x_ghost(xshape[0], 0);
  std::fill(std::next(x_ghost.begin(), xmap->size_local()), x_ghost.end(), 1);
  add_mesh(geometry.x(), xshape, geometry.input_global_indices(), x_ghost,
           cells, cshape, *topology.index_map(tdim), topology.cell_type(),
           topology.dim(), piece_node);

  // Create filepath for a .vtu file
  auto create_vtu_path = [file_root = _filename.parent_path(),
                          file_name = _filename.stem(), counter_str](int rank)
  {
    std::filesystem::path vtu = file_root / file_name;
    vtu += +"_p" + std::to_string(rank) + "_" + counter_str;
    vtu.replace_extension("vtu");
    return vtu;
  };

  // Save VTU XML to file
  const int mpi_rank = dolfinx::MPI::rank(_comm.comm());
  std::filesystem::path vtu = create_vtu_path(mpi_rank);
  if (vtu.has_parent_path())
    std::filesystem::create_directories(vtu.parent_path());
  xml_vtu.save_file(vtu.c_str(), "  ");

  // Create a PVTU XML object on rank 0
  std::filesystem::path p_pvtu = _filename.parent_path() / _filename.stem();
  p_pvtu += counter_str;
  p_pvtu.replace_extension("pvtu");
  if (mpi_rank == 0)
  {
    pugi::xml_document xml_pvtu;
    pugi::xml_node vtk_node = xml_pvtu.append_child("VTKFile");
    vtk_node.append_attribute("type") = "PUnstructuredGrid";
    vtk_node.append_attribute("version") = "1.0";
    pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
    grid_node.append_attribute("GhostLevel") = 1;

    // Add mesh metadata to PVTU object
    add_pvtu_mesh(grid_node);

    // Add data for each process to the PVTU object
    const int mpi_size = dolfinx::MPI::size(_comm.comm());
    for (int r = 0; r < mpi_size; ++r)
    {
      std::filesystem::path vtu = create_vtu_path(r);
      pugi::xml_node piece_node = grid_node.append_child("Piece");
      piece_node.append_attribute("Source") = vtu.filename().c_str();
    }
    // Write PVTU file
    if (p_pvtu.has_parent_path())
      std::filesystem::create_directories(p_pvtu.parent_path());
    xml_pvtu.save_file(p_pvtu.c_str(), "  ");
  }

  // Append PVD file
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file") = p_pvtu.filename().c_str();
}
//----------------------------------------------------------------------------
void io::VTKFile::write_functions(
    const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
    double time)
{
  write_function(u, time, _pvd_xml, _filename);
}
//----------------------------------------------------------------------------
void io::VTKFile::write_functions(
    const std::vector<
        std::reference_wrapper<const fem::Function<std::complex<double>>>>& u,
    double time)
{
  write_function(u, time, _pvd_xml, _filename);
}
//----------------------------------------------------------------------------
