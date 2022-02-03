// Copyright (C) 2005-2020 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKFile.h"
#include "cells.h"
#include "pugixml.hpp"
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
#include <sstream>
#include <string>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xcomplex.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

using namespace dolfinx;

namespace
{
/// Tabulate the coordinate for every 'node' in a Lagrange function
/// space.
/// @param[in] V The function space. Must be a (discontinuous) Lagrange
/// space.
/// @return An array with shape (num_dofs, 3) array where the ith row
/// corresponds to the coordinate of the ith dof in `V` (local to
/// process)
/// @pre `V` must be Lagrange and must not be a subspace
xt::xtensor<double, 2>
tabulate_lagrange_dof_coordinates(const dolfinx::fem::FunctionSpace& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // Get dofmap data
  auto dofmap = V.dofmap();
  assert(dofmap);
  auto map_dofs = dofmap->index_map;
  assert(map_dofs);
  const int index_map_bs = dofmap->index_map_bs();
  const int dofmap_bs = dofmap->bs();

  // Get element data
  auto element = V.element();
  assert(element);
  const int e_bs = element->block_size();
  const std::size_t scalar_dofs = element->space_dimension() / e_bs;
  const std::int32_t num_nodes
      = index_map_bs * (map_dofs->size_local() + map_dofs->num_ghosts())
        / dofmap_bs;

  // Get the dof coordinates on the reference element and the  mesh
  // coordinate map
  const xt::xtensor<double, 2>& X = element->interpolation_points();
  const fem::CoordinateElement& cmap = mesh->geometry().cmap();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& dofmap_x
      = mesh->geometry().dofmap();
  xtl::span<const double> x_g = mesh->geometry().x();
  const std::size_t num_dofs_g = dofmap_x.num_links(0);

  xtl::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }
  const auto apply_dof_transformation
      = element->get_dof_transformation_function<double>();

  // Tabulate basis functions at node reference coordinates
  const xt::xtensor<double, 2> phi
      = xt::view(cmap.tabulate(0, X), 0, xt::all(), xt::all(), 0);

  // Loop over cells and tabulate dofs
  auto map = mesh->topology().index_map(tdim);
  assert(map);
  const std::int32_t num_cells = map->size_local() + map->num_ghosts();
  xt::xtensor<double, 2> x = xt::zeros<double>({scalar_dofs, gdim});
  xt::xtensor<double, 2> coordinate_dofs({num_dofs_g, gdim});
  xt::xtensor<double, 2> coords = xt::zeros<double>({num_nodes, 3});
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    // Extract cell geometry
    auto dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs_x.size(); ++i)
    {
      std::copy_n(std::next(x_g.begin(), 3 * dofs_x[i]), gdim,
                  std::next(coordinate_dofs.begin(), i * gdim));
    }

    // Tabulate dof coordinates on cell
    cmap.push_forward(x, coordinate_dofs, phi);
    apply_dof_transformation(xtl::span(x.data(), x.size()),
                             xtl::span(cell_info.data(), cell_info.size()), c,
                             x.shape(1));

    // Copy dof coordinates into vector
    auto dofs = dofmap->cell_dofs(c);
    for (std::size_t i = 0; i < dofs.size(); ++i)
    {
      std::int32_t dof = dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coords(dof, j) = x(i, j);
    }
  }

  return coords;
}
//-----------------------------------------------------------------------------

/// Given a FunctionSpace, create a topology and geometry based on the
/// dof coordinates.
/// @note Only supports (discontinuous) Lagrange functions
/// @param[in] u The function
std::pair<xt::xtensor<double, 2>, xt::xtensor<std::int64_t, 2>>
vtk_mesh_from_space(const fem::FunctionSpace& V)
{
  auto mesh = V.mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();

  assert(V.element());
  if (V.element()->is_mixed())
    throw std::runtime_error("Can create VTK mesh from a mixed element");

  xt::xtensor<double, 2> x = tabulate_lagrange_dof_coordinates(V);
  auto map = mesh->topology().index_map(tdim);
  const std::size_t num_cells = map->size_local() + map->num_ghosts();

  // Create permutation from DOLFINx dof ordering to VTK
  auto dofmap = V.dofmap();
  assert(dofmap);
  const std::uint32_t num_nodes = dofmap->cell_dofs(0).size();
  std::vector<std::uint8_t> vtkmap = dolfinx::io::cells::transpose(
      io::cells::perm_vtk(mesh->topology().cell_type(), num_nodes));

  // Extract topology for all local cells as
  // [N0, v0_0, ...., v0_N0, N1, v1_0, ...., v1_N1, ....]
  // std::vector<std::int64_t> vtk_topology(num_cells * (num_nodes + 1));
  xt::xtensor<std::int64_t, 2> vtk_topology({num_cells, num_nodes});
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap->cell_dofs(c);
    for (std::size_t i = 0; i < dofs.size(); ++i)
      vtk_topology(c, i) = dofs[vtkmap[i]];
  }

  return {std::move(x), std::move(vtk_topology)};
}
//-----------------------------------------------------------------------------

/// Extract the cell topology (connectivity) in VTK ordering for all
/// cells the mesh. The 'topology' includes higher-order 'nodes'. The
/// index of a 'node' corresponds to the index of DOLFINx geometry
/// 'nodes'.
/// @param [in] mesh The mesh
/// @return The cell topology in VTK ordering and in term of the DOLFINx
/// geometry 'nodes'
/// @note The indices in the return array correspond to the point
/// indices in the mesh geometry array
/// @note Even if the indices are local (int32), both Fides and VTX
/// require int64 as local input
xt::xtensor<std::int64_t, 2> extract_vtk_connectivity(const mesh::Mesh& mesh)
{
  // Get DOLFINx to VTK permutation
  // FIXME: Use better way to get number of nodes
  const graph::AdjacencyList<std::int32_t>& dofmap_x = mesh.geometry().dofmap();
  const std::size_t num_nodes = dofmap_x.num_links(0);
  std::vector map = dolfinx::io::cells::transpose(
      dolfinx::io::cells::perm_vtk(mesh.topology().cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (mesh.topology().cell_type() == dolfinx::mesh::CellType::hexahedron
      and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }
  // Extract mesh 'nodes'
  const int tdim = mesh.topology().dim();
  const std::size_t num_cells = mesh.topology().index_map(tdim)->size_local();

  // Build mesh connectivity

  // Loop over cells
  xt::xtensor<std::int64_t, 2> topology({num_cells, num_nodes});
  for (std::size_t c = 0; c < num_cells; ++c)
  {
    // For each cell, get the 'nodes' and place in VTK order
    auto dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs_x.size(); ++i)
      topology(c, i) = dofs_x[map[i]];
  }

  return topology;
}
//-----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/// Return true if Function is a cell-wise constant, otherwise false
bool is_cellwise(const fem::FunctionSpace& V)
{
  assert(V.element());
  const int rank = V.element()->value_shape().size();
  assert(V.mesh());
  const int tdim = V.mesh()->topology().dim();
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
  const int num_digits = 6;
  std::string counter = std::to_string(n);
  return std::string(num_digits - counter.size(), '0').append(counter);
}
//----------------------------------------------------------------------------

/// Get the VTK cell type integer
std::int8_t get_vtk_cell_type(mesh::CellType cell, int dim)
{
  if (cell == mesh::CellType::prism and dim == 2)
    throw std::runtime_error("More work needed for prism cell");

  // Get cell type
  mesh::CellType cell_type = mesh::cell_entity_type(cell, dim, 0);

  // Determine VTK cell type (arbitrary Lagrange elements)
  // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
  switch (cell_type)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return 68;
  case mesh::CellType::triangle:
    return 69;
  case mesh::CellType::quadrilateral:
    return 70;
  case mesh::CellType::tetrahedron:
    return 71;
  case mesh::CellType::hexahedron:
    return 72;
  default:
    throw std::runtime_error("Unknown cell type");
  }
}
//----------------------------------------------------------------------------

/// Convert an xtensor to a std::string
template <typename T>
std::string xt_to_string(const T& x, int precision)
{
  std::stringstream s;
  s.precision(precision);
  std::for_each(x.begin(), x.end(), [&s](auto e) { s << e << " "; });
  return s.str();
}
//----------------------------------------------------------------------------

void add_pvtu_mesh(pugi::xml_node& node)
{
  pugi::xml_node vertex_data_node = node.append_child("PPoints");
  pugi::xml_node data_node = vertex_data_node.append_child("PDataArray");
  data_node.append_attribute("type") = "Float64";
  data_node.append_attribute("NumberOfComponents") = "3";
}
//----------------------------------------------------------------------------
/// At data to a pugixml node
template <typename Scalar>
void add_data(const std::string& name, int rank,
              const xtl::span<const Scalar>& values, pugi::xml_node& data_node)
{
  pugi::xml_node field_node = data_node.append_child("DataArray");
  field_node.append_attribute("type") = "Float64";
  field_node.append_attribute("Name") = name.c_str();
  field_node.append_attribute("format") = "ascii";

  if (rank == 1)
    field_node.append_attribute("NumberOfComponents") = 3;
  else if (rank == 2)
    field_node.append_attribute("NumberOfComponents") = 9;
  field_node.append_child(pugi::node_pcdata)
      .set_value(xt_to_string(values, 16).c_str());
}
//----------------------------------------------------------------------------

/// At mesh geometry and topology data to a pugixml node. The function
/// adds the Points and Cells nodes to the input node/
void add_mesh(const xt::xtensor<double, 2>& x,
              const xt::xtensor<std::int64_t, 2>& cells,
              mesh::CellType celltype, int tdim, pugi::xml_node& piece_node)
{
  // Add geometry (points)

  pugi::xml_node points_node = piece_node.append_child("Points");
  pugi::xml_node x_node = points_node.append_child("DataArray");
  x_node.append_attribute("type") = "Float64";
  x_node.append_attribute("NumberOfComponents") = "3";
  x_node.append_attribute("format") = "ascii";
  x_node.append_child(pugi::node_pcdata).set_value(xt_to_string(x, 16).c_str());

  // Add topology (cells)

  pugi::xml_node cells_node = piece_node.append_child("Cells");
  pugi::xml_node connectivity_node = cells_node.append_child("DataArray");
  connectivity_node.append_attribute("type") = "Int32";
  connectivity_node.append_attribute("Name") = "connectivity";
  connectivity_node.append_attribute("format") = "ascii";

  // Extra cell topology
  // xt::xtensor<std::int64_t, 2> cells = extract_vtk_connectivity(mesh);
  std::stringstream ss;
  std::for_each(cells.begin(), cells.end(), [&ss](auto& v) { ss << v << " "; });
  connectivity_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());

  pugi::xml_node offsets_node = cells_node.append_child("DataArray");
  offsets_node.append_attribute("type") = "Int32";
  offsets_node.append_attribute("Name") = "offsets";
  offsets_node.append_attribute("format") = "ascii";
  std::stringstream ss_offset;
  int num_nodes = cells.shape(1);
  for (std::size_t i = 0; i < cells.shape(0); ++i)
    ss_offset << (i + 1) * num_nodes << " ";
  offsets_node.append_child(pugi::node_pcdata)
      .set_value(ss_offset.str().c_str());

  pugi::xml_node type_node = cells_node.append_child("DataArray");
  type_node.append_attribute("type") = "Int8";
  type_node.append_attribute("Name") = "types";
  type_node.append_attribute("format") = "ascii";
  int vtk_celltype = get_vtk_cell_type(celltype, tdim);
  std::stringstream s;
  for (std::size_t c = 0; c < cells.shape(0); ++c)
    s << vtk_celltype << " ";
  type_node.append_child(pugi::node_pcdata).set_value(s.str().c_str());
}
//----------------------------------------------------------------------------
template <typename Scalar>
void write_function(
    const std::vector<std::reference_wrapper<const fem::Function<Scalar>>>& u,
    double time, std::unique_ptr<pugi::xml_document>& xml_doc,
    const std::filesystem::path& filename)
{
  if (!xml_doc)
    throw std::runtime_error("VTKFile has been closed");
  if (u.empty())
    return;

  // TODO: check elements are compatible (same point data element, plus
  // possible cell data elements)

  // TODO: Check for sub-elements (dis-allow?)

  // Extract first function space with pointwise data
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

  // Check compatibility all functions
  auto mesh0 = V0->mesh();
  assert(mesh0);
  auto element0 = V0->element();
  for (auto& v : u)
  {
    auto V = v.get().function_space();

    // Check that functions share common mesh
    if (V->mesh() != mesh0)
    {
      throw std::runtime_error(
          "All functions written to VTK file must share the same Mesh.");
    }

    // Check for sub-functions
    if (!V->component().empty())
      throw std::runtime_error("Cannot write sub-Functions to VTK file.");

    // Check that pointwise element are the same (up to the block size)
    if (!is_cellwise(*V))
    {
      if (*(V->element()) != *element0)
      {
        throw std::runtime_error(
            "All functions written to VTK file must have same element.");
      }
    }
  }

  const MPI_Comm comm = mesh0->comm();
  const int mpi_rank = dolfinx::MPI::rank(comm);

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
  vtk_node_vtu.append_attribute("version") = "0.1";
  pugi::xml_node grid_node_vtu = vtk_node_vtu.append_child("UnstructuredGrid");

  // Build mesh data to match first Function
  xt::xtensor<double, 2> x;
  xt::xtensor<std::int64_t, 2> cells;
  if (is_cellwise(*V0))
  {
    cells = extract_vtk_connectivity(*mesh0);
    const mesh::Geometry& geometry = mesh0->geometry();
    x = xt::adapt(geometry.x().data(), geometry.x().size(), xt::no_ownership(),
                  std::vector({geometry.x().size() / 3, std::size_t(3)}));
  }
  else
    std::tie(x, cells) = vtk_mesh_from_space(*V0);

  // Add "Piece" node and required metadata
  pugi::xml_node piece_node = grid_node_vtu.append_child("Piece");
  piece_node.append_attribute("NumberOfPoints") = x.shape(0);
  piece_node.append_attribute("NumberOfCells") = cells.shape(0);

  // Add mesh data to "Piece" node
  add_mesh(x, cells, mesh0->topology().cell_type(), mesh0->topology().dim(),
           piece_node);

  // FIXME: is this actually setting the first?
  // Set last scalar/vector/tensor Functions in u to be the 'active'
  // (default) field(s)
  constexpr std::array tensor_str = {"Scalars", "Vectors", "Tensors"};
  for (auto _u : u)
  {
    auto V = _u.get().function_space();
    assert(V);
    std::string data_type = is_cellwise(*V) ? "CellData" : "PointData";
    if (piece_node.child(data_type.c_str()).empty())
      piece_node.append_child(data_type.c_str());

    const int rank = V->element()->value_shape().size();
    pugi::xml_node data_node = piece_node.child(data_type.c_str());
    if (data_node.attribute(tensor_str[rank]).empty())
      data_node.append_attribute(tensor_str[rank]);
    pugi::xml_attribute data = data_node.attribute(tensor_str[rank]);
    data.set_value(_u.get().name.c_str());
  }

  // Add cell/point data to VTU node
  for (auto _u : u)
  {
    auto V = _u.get().function_space();
    assert(V);
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

      std::vector<Scalar> data(cells.shape(0) * num_comp, 0);
      auto u_vector = _u.get().x()->array();
      for (std::size_t c = 0; c < cells.shape(0); ++c)
      {
        xtl::span<const std::int32_t> dofs = dofmap->cell_dofs(c);
        for (std::size_t i = 0; i < dofs.size(); ++i)
          for (int k = 0; k < bs; ++k)
            data[num_comp * c + k] = u_vector[bs * dofs[i] + k];
      }

      add_data(_u.get().name, rank, xtl::span<const Scalar>(data), data_node);
    }
    else
    {
      // -- Point-wise data

      pugi::xml_node data_node = piece_node.child("PointData");
      assert(!data_node.empty());

      auto pad_data = [num_comp](const fem::FunctionSpace& V,
                                 const xtl::span<const Scalar>& u, int rank)
      {
        auto dofmap = V.dofmap();
        int bs = dofmap->bs();
        auto map = dofmap->index_map;
        int map_bs = dofmap->index_map_bs();
        std::int32_t num_dofs_block
            = map_bs * (map->size_local() + map->num_ghosts()) / bs;

        std::vector<Scalar> data(num_dofs_block * num_comp, 0);
        for (int i = 0; i < num_dofs_block; ++i)
        {
          std::copy_n(std::next(u.cbegin(), i * map_bs), map_bs,
                      std::next(data.begin(), i * num_comp));
        }

        return data;
      };

      if (V == V0)
      {
        // Identical spaces
        if (mesh0->geometry().dim() == 3)
          add_data(_u.get().name, rank, _u.get().x()->array(), data_node);
        else
        {
          // Pad with zeros and then add
          auto data = pad_data(*V, _u.get().x()->array(), rank);
          add_data(_u.get().name, rank, xtl::span<const Scalar>(data),
                   data_node);
        }
      }
      else if (*element == *element0)
      {
        // Same element, possibly different dofmaps
        // TODO: we need dofmap0 to be 'blocked'
        // TODO: check ElementDofLayout?

        // Get dofmaps
        auto dofmap0 = V0->dofmap();
        assert(dofmap0);
        auto dofmap = V->dofmap();
        assert(dofmap);

        auto map = dofmap->index_map;
        int map_bs = dofmap->index_map_bs();
        int bs = dofmap->bs();
        int bs0 = dofmap0->bs();
        std::int32_t num_dofs_block
            = map_bs * (map->size_local() + map->num_ghosts()) / bs;

        // Interpolate on each cell
        auto u_vector = _u.get().x()->array();
        std::vector<Scalar> u(u_vector.size());
        for (std::size_t c = 0; c < cells.shape(0); ++c)
        {
          xtl::span<const std::int32_t> dofs0 = dofmap0->cell_dofs(c);
          xtl::span<const std::int32_t> dofs = dofmap->cell_dofs(c);
          assert(bs0 * dofs0.size() == bs * dofs.size());
          for (std::size_t i = 0; i < dofs0.size(); ++i)
          {
            for (int k = 0; k < bs; ++k)
            {
              int index = bs * i + k;
              std::div_t dv0 = std::div(index, bs0);
              u[bs0 * dofs0[dv0.quot] + dv0.rem] = u_vector[bs * dofs[i] + k];
            }
          }
        }

        // Pack/add data
        if (mesh0->geometry().dim() == 3)
          add_data(_u.get().name, rank, xtl::span<const Scalar>(u), data_node);
        else
        {
          // Pad with zeros and then add
          auto data = pad_data(*V, _u.get().x()->array(), rank);
          add_data(_u.get().name, rank, xtl::span<const Scalar>(data),
                   data_node);
        }
      }
      else
        throw std::runtime_error(
            "Elements differ, not permitted for VTK output");
    }
  }

  // Save VTU XML to file
  std::filesystem::path vtu = filename.parent_path();
  if (!vtu.empty())
    vtu += "/";
  vtu += filename.stem().string() + "_p" + std::to_string(mpi_rank) + "_"
         + counter_str;
  vtu.replace_extension("vtu");
  xml_vtu.save_file(vtu.c_str(), "  ");

  // Create a PVTU XML object on rank 0
  std::filesystem::path p_pvtu = filename.parent_path();
  if (!p_pvtu.empty())
    p_pvtu += "/";
  p_pvtu += filename.stem().string() + counter_str;
  p_pvtu.replace_extension("pvtu");
  if (mpi_rank == 0)
  {
    pugi::xml_document xml_pvtu;
    pugi::xml_node vtk_node = xml_pvtu.append_child("VTKFile");
    vtk_node.append_attribute("type") = "PUnstructuredGrid";
    vtk_node.append_attribute("version") = "0.1";
    pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
    grid_node.append_attribute("GhostLevel") = 0;
    for (auto _u : u)
    {
      assert(_u.get().function_space());
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

    // Add field data
    std::vector<std::string> components = {""};
    if constexpr (!std::is_scalar<Scalar>::value)
      components = {"real", "imag"};

    for (auto _u : u)
    {
      auto V = _u.get().function_space();
      assert(V);
      std::string d_type = is_cellwise(*V) ? "PCellData" : "PPointData";
      pugi::xml_node data_pnode = grid_node.child(d_type.c_str());
      const int rank = V->element()->value_shape().size();
      int ncomps = 0;
      if (rank == 1)
        ncomps = 3;
      else if (rank == 2)
        ncomps = 9;
      for (const auto& component : components)
      {
        pugi::xml_node data_node = data_pnode.append_child("PDataArray");
        data_node.append_attribute("type") = "Float64";
        if constexpr (!std::is_scalar<Scalar>::value)
        {
          data_node.append_attribute("Name")
              = (component + "_" + _u.get().name).c_str();
        }
        else
        {
          data_node.append_attribute("Name")
              = (component + "" + _u.get().name).c_str();
        }

        data_node.append_attribute("NumberOfComponents") = ncomps;
      }

      // Add data for each process to the PVTU object
      const int mpi_size = dolfinx::MPI::size(comm);
      for (int i = 0; i < mpi_size; ++i)
      {
        std::filesystem::path vtu = filename.stem();
        vtu += "_p" + std::to_string(i) + "_" + counter_str;
        vtu.replace_extension("vtu");
        pugi::xml_node piece_node = grid_node.append_child("Piece");
        piece_node.append_attribute("Source")
            = vtu.stem().replace_extension("vtu").c_str();
      }
    }

    // Write PVTU file
    xml_pvtu.save_file(p_pvtu.c_str(), "  ");
  }

  // Append PVD file
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file")
      = p_pvtu.stem().replace_extension("pvtu").c_str();
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
  vtk_node.append_attribute("version") = "0.1";
  vtk_node.append_child("Collection");
}
//----------------------------------------------------------------------------
io::VTKFile::~VTKFile()
{
  if (_pvd_xml and MPI::rank(_comm.comm()) == 0)
    _pvd_xml->save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void io::VTKFile::close()
{
  if (_pvd_xml and MPI::rank(_comm.comm()) == 0)
  {
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
  if (!_pvd_xml and MPI::rank(_comm.comm()) == 0)
    throw std::runtime_error("VTKFile has already been closed");

  if (MPI::rank(_comm.comm()) == 0)
    _pvd_xml->save_file(_filename.c_str(), "  ");
}
//----------------------------------------------------------------------------
void io::VTKFile::write(const mesh::Mesh& mesh, double time)
{
  if (!_pvd_xml)
    throw std::runtime_error("VTKFile has already been closed");

  const int mpi_rank = MPI::rank(_comm.comm());

  // Get the PVD "Collection" node
  pugi::xml_node xml_collections
      = _pvd_xml->child("VTKFile").child("Collection");
  assert(xml_collections);

  // Compute counter string
  const std::string counter_str = get_counter(xml_collections, "DataSet");

  // Get mesh data for this rank
  const mesh::Topology& topology = mesh.topology();
  const mesh::Geometry& geometry = mesh.geometry();
  const int tdim = topology.dim();
  const std::int32_t num_points
      = geometry.index_map()->size_local() + geometry.index_map()->num_ghosts();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local();

  // Create a VTU XML object
  pugi::xml_document xml_vtu;
  pugi::xml_node vtk_node_vtu = xml_vtu.append_child("VTKFile");
  vtk_node_vtu.append_attribute("type") = "UnstructuredGrid";
  vtk_node_vtu.append_attribute("version") = "0.1";
  pugi::xml_node grid_node_vtu = vtk_node_vtu.append_child("UnstructuredGrid");

  // Add "Piece" node and required metadata
  pugi::xml_node piece_node = grid_node_vtu.append_child("Piece");
  piece_node.append_attribute("NumberOfPoints") = num_points;
  piece_node.append_attribute("NumberOfCells") = num_cells;

  // Add mesh data to "Piece" node
  xt::xtensor<std::int64_t, 2> cells = extract_vtk_connectivity(mesh);
  xt::xtensor<double, 2> x
      = xt::adapt(geometry.x().data(), geometry.x().size(), xt::no_ownership(),
                  std::vector({geometry.x().size() / 3, std::size_t(3)}));
  add_mesh(x, cells, topology.cell_type(), topology.dim(), piece_node);

  // Save VTU XML to file
  std::filesystem::path vtu = _filename.parent_path();
  if (!vtu.empty())
    vtu += "/";
  vtu += _filename.stem().string() + "_p" + std::to_string(mpi_rank) + "_"
         + counter_str;
  vtu.replace_extension("vtu");
  xml_vtu.save_file(vtu.c_str(), "  ");

  // Create a PVTU XML object on rank 0
  std::filesystem::path p_pvtu = _filename.parent_path();
  if (!p_pvtu.empty())
    p_pvtu += "/";
  p_pvtu += _filename.stem().string() + counter_str;
  p_pvtu.replace_extension("pvtu");
  if (mpi_rank == 0)
  {
    pugi::xml_document xml_pvtu;
    pugi::xml_node vtk_node = xml_pvtu.append_child("VTKFile");
    vtk_node.append_attribute("type") = "PUnstructuredGrid";
    vtk_node.append_attribute("version") = "0.1";
    pugi::xml_node grid_node = vtk_node.append_child("PUnstructuredGrid");
    grid_node.append_attribute("GhostLevel") = 0;

    // Add mesh metadata to PVTU object
    add_pvtu_mesh(grid_node);

    // Add data for each process to the PVTU object
    const int mpi_size = MPI::size(_comm.comm());
    for (int i = 0; i < mpi_size; ++i)
    {
      std::filesystem::path vtu = _filename.stem();
      vtu += "_p" + std::to_string(i) + "_" + counter_str;
      vtu.replace_extension("vtu");
      pugi::xml_node piece_node = grid_node.append_child("Piece");
      piece_node.append_attribute("Source") = vtu.c_str();
    }

    // Write PVTU file
    xml_pvtu.save_file(p_pvtu.c_str(), "  ");
  }

  // Append PVD file
  pugi::xml_node dataset_node = xml_collections.append_child("DataSet");
  dataset_node.append_attribute("timestep") = time;
  dataset_node.append_attribute("part") = "0";
  dataset_node.append_attribute("file")
      = p_pvtu.stem().replace_extension("pvtu").c_str();
}
//----------------------------------------------------------------------------
void io::VTKFile::write(
    const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
    double time)
{
  write_function(u, time, _pvd_xml, _filename);
}
//----------------------------------------------------------------------------
void io::VTKFile::write(const std::vector<std::reference_wrapper<
                            const fem::Function<std::complex<double>>>>& /*u*/,
                        double /*time*/)
{
  // write_function(u, time, _pvd_xml, _filename);
}
//----------------------------------------------------------------------------
