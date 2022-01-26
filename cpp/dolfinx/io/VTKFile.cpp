// Copyright (C) 2005-2020 Garth N. Wells and Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "VTKFile.h"
#include "cells.h"
#include "pugixml.hpp"
#include "xdmf_utils.h"
#include <boost/filesystem.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <sstream>
#include <string>
#include <xtensor/xcomplex.hpp>
#include <xtl/xspan.hpp>

using namespace dolfinx;

namespace
{
//----------------------------------------------------------------------------
/// Return true if Function is a cell-wise constant, otherwise false
template <typename Scalar>
bool _is_cellwise(const fem::Function<Scalar>& u)
{
  assert(u.function_space());

  assert(u.function_space()->element());
  const int rank = u.function_space()->element()->value_shape().size();
  assert(u.function_space()->mesh());
  const int tdim = u.function_space()->mesh()->topology().dim();
  int cell_based_dim = 1;
  for (int i = 0; i < rank; ++i)
    cell_based_dim *= tdim;

  assert(u.function_space()->dofmap());
  if (u.function_space()->dofmap()->element_dof_layout().num_dofs()
      == cell_based_dim)
  {
    return true;
  }
  else
    return false;
}
//----------------------------------------------------------------------------
bool is_cellwise(const fem::Function<double>& u) { return _is_cellwise(u); }
//----------------------------------------------------------------------------
bool is_cellwise(const fem::Function<std::complex<double>>& u)
{
  return _is_cellwise(u);
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

/// Convert an xtensor to a std::string
template <typename T>
std::string xt_to_string(const T& x, int precision)
{
  std::stringstream s;
  s.precision(precision);
  std::for_each(x.begin(), x.end(), [&s](auto e) { s << e << " "; });
  return s.str();
}

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
void _add_data(const fem::Function<Scalar>& u,
               const xt::xtensor<Scalar, 2>& values, pugi::xml_node& data_node)
{
  const int rank = u.function_space()->element()->value_shape().size();
  const int dim = u.function_space()->element()->value_size();
  if (rank == 1)
  {
    if (!(dim == 2 or dim == 3))
    {
      throw std::runtime_error(
          "Cannot write data to VTK file. Don't know how to handle vector "
          "function with dimension other than 2 or 3");
    }
  }
  else if (rank == 2)
  {
    if (!(dim == 4 or dim == 9))
    {
      throw std::runtime_error(
          "Cannot write data to VTK file. Don't know how to handle tensor "
          "function with dimension other than 4 or 9");
    }
  }
  else if (rank > 2)
  {
    throw std::runtime_error(
        "Cannot write data to VTK file. "
        "Only scalar, vector and tensor functions can be saved in VTK format");
  }
  // Loop for complex numbers, saved as real and imaginary part
  std::vector<std::string> components = {""};
  if constexpr (!std::is_scalar<Scalar>::value)
    components = {"real", "imag"};

  for (const auto& component : components)
  {
    if constexpr (!std::is_scalar<Scalar>::value)
    {
      pugi::xml_node field_node = data_node.append_child("DataArray");
      field_node.append_attribute("type") = "Float64";
      field_node.append_attribute("Name") = (component + "_" + u.name).c_str();
      field_node.append_attribute("format") = "ascii";
      xt::xtensor<double, 2> values_comp({values.shape()});

      if (component == "real")
        values_comp = xt::real(values);
      else if (component == "imag")
        values_comp = xt::imag(values);
      if (rank == 0)
      {
        field_node.append_child(pugi::node_pcdata)
            .set_value(xt_to_string(values_comp, 16).c_str());
      }
      else if (rank == 1)
      {
        field_node.append_attribute("NumberOfComponents") = 3;
        if (dim == 2)
        {
          assert(values_comp.shape(1) == 2);
          std::stringstream ss;
          for (std::size_t i = 0; i < values_comp.shape(0); ++i)
          {
            for (int j = 0; j < 2; ++j)
              ss << values_comp(i, j) << " ";
            ss << 0.0 << " ";
          }
          field_node.append_child(pugi::node_pcdata)
              .set_value(ss.str().c_str());
        }
        else
        {
          assert(values_comp.shape(2) == 3);
          field_node.append_child(pugi::node_pcdata)
              .set_value(xt_to_string(values_comp, 16).c_str());
        }
      }
      else if (rank == 2)
      {
        field_node.append_attribute("NumberOfComponents") = 9;
        if (dim == 4)
        {
          // Pad 2D tensors with 0.0 to make them 3D
          std::stringstream ss;
          for (std::size_t i = 0; i < values_comp.shape(0); ++i)
          {
            for (int j = 0; j < 2; ++j)
            {
              ss << values_comp(i, (2 * j + 0)) << " ";
              ss << values_comp(i, (2 * j + 1)) << " ";
              ss << 0.0 << " ";
            }
            ss << 0.0 << " ";
            ss << 0.0 << " ";
            ss << 0.0 << "  ";
          }
          field_node.append_child(pugi::node_pcdata)
              .set_value(ss.str().c_str());
        }
        else
        {
          field_node.append_child(pugi::node_pcdata)
              .set_value(xt_to_string(values_comp, 16).c_str());
        }
      }
    }
    else
    {
      pugi::xml_node field_node = data_node.append_child("DataArray");
      field_node.append_attribute("type") = "Float64";
      field_node.append_attribute("Name") = (component + u.name).c_str();
      field_node.append_attribute("format") = "ascii";

      if (rank == 0)
      {
        field_node.append_child(pugi::node_pcdata)
            .set_value(xt_to_string(values, 16).c_str());
      }
      else if (rank == 1)
      {
        field_node.append_attribute("NumberOfComponents") = 3;
        if (dim == 2)
        {
          assert(values.shape(1) == 2);
          std::stringstream ss;
          for (size_t i = 0; i < values.shape(0); ++i)
          {
            for (int j = 0; j < 2; ++j)
              ss << values(i, j) << " ";
            ss << 0.0 << " ";
          }
          field_node.append_child(pugi::node_pcdata)
              .set_value(ss.str().c_str());
        }
        else
        {
          assert(values.shape(1) == 3);
          field_node.append_child(pugi::node_pcdata)
              .set_value(xt_to_string(values, 16).c_str());
        }
      }
      else if (rank == 2)
      {
        field_node.append_attribute("NumberOfComponents") = 9;
        if (dim == 4)
        {
          // Pad 2D tensors with 0.0 to make them 3D
          std::stringstream ss;
          for (size_t i = 0; i < values.shape(0); ++i)
          {
            for (int j = 0; j < 2; ++j)
            {
              ss << values(i, (2 * j + 0)) << " ";
              ss << values(i, (2 * j + 1)) << " ";
              ss << 0.0 << " ";
            }
            ss << 0.0 << " ";
            ss << 0.0 << " ";
            ss << 0.0 << "  ";
          }
          field_node.append_child(pugi::node_pcdata)
              .set_value(ss.str().c_str());
        }
        else
        {
          field_node.append_child(pugi::node_pcdata)
              .set_value(xt_to_string(values, 16).c_str());
        }
      }
    }
  }
}
//----------------------------------------------------------------------------
void add_data(const fem::Function<double>& u,
              const xt::xtensor<double, 2>& values, pugi::xml_node& data_node)
{
  _add_data(u, values, data_node);
}
//----------------------------------------------------------------------------
void add_data(const fem::Function<std::complex<double>>& u,
              const xt::xtensor<std::complex<double>, 2>& values,
              pugi::xml_node& data_node)
{
  _add_data(u, values, data_node);
}
//----------------------------------------------------------------------------

/// At mesh geometry and topology data to a pugixml node. The function /
/// adds the Points and Cells nodes to the input node/
void add_mesh(const mesh::Mesh& mesh, pugi::xml_node& piece_node)
{
  const mesh::Topology& topology = mesh.topology();
  const mesh::Geometry& geometry = mesh.geometry();
  const int tdim = topology.dim();
  const std::int32_t num_cells = topology.index_map(tdim)->size_local();

  // Add geometry (points)

  pugi::xml_node points_node = piece_node.append_child("Points");
  pugi::xml_node x_node = points_node.append_child("DataArray");
  x_node.append_attribute("type") = "Float64";
  x_node.append_attribute("NumberOfComponents") = "3";
  x_node.append_attribute("format") = "ascii";
  const auto x
      = xt::adapt(geometry.x().data(), geometry.x().size(), xt::no_ownership(),
                  std::vector({geometry.x().size() / 3, std::size_t(3)}));
  x_node.append_child(pugi::node_pcdata).set_value(xt_to_string(x, 16).c_str());

  // Add topology(cells)

  pugi::xml_node cells_node = piece_node.append_child("Cells");
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();

  pugi::xml_node connectivity_node = cells_node.append_child("DataArray");
  connectivity_node.append_attribute("type") = "Int32";
  connectivity_node.append_attribute("Name") = "connectivity";
  connectivity_node.append_attribute("format") = "ascii";

  // Get map from VTK index i to DOLFIN index j
  int num_nodes = geometry.cmap().create_dof_layout().num_dofs();

  std::vector<std::uint8_t> map = io::cells::transpose(
      io::cells::perm_vtk(topology.cell_type(), num_nodes));
  // TODO: Remove when when paraview issue 19433 is resolved
  // (https://gitlab.kitware.com/paraview/paraview/issues/19433)
  if (topology.cell_type() == mesh::CellType::hexahedron and num_nodes == 27)
  {
    map = {0,  9, 12, 3,  1, 10, 13, 4,  18, 15, 21, 6,  19, 16,
           22, 7, 2,  11, 5, 14, 8,  17, 20, 23, 24, 25, 26};
  }

  std::stringstream ss;
  for (int c = 0; c < x_dofmap.num_nodes(); ++c)
  {
    xtl::span<const std::int32_t> cell = x_dofmap.links(c);
    const int num_cell_dofs = cell.size();
    for (int i = 0; i < num_cell_dofs; ++i)
      ss << cell[map[i]] << " ";
  }
  connectivity_node.append_child(pugi::node_pcdata).set_value(ss.str().c_str());

  pugi::xml_node offsets_node = cells_node.append_child("DataArray");
  offsets_node.append_attribute("type") = "Int32";
  offsets_node.append_attribute("Name") = "offsets";
  offsets_node.append_attribute("format") = "ascii";
  const std::vector<std::int32_t>& offsets = x_dofmap.offsets();
  std::stringstream ss_offset;
  ss_offset.precision(0);
  for (std::int32_t i = 1; i <= num_cells; ++i)
    ss_offset << offsets[i] << " ";

  offsets_node.append_child(pugi::node_pcdata)
      .set_value(ss_offset.str().c_str());

  pugi::xml_node type_node = cells_node.append_child("DataArray");
  type_node.append_attribute("type") = "Int8";
  type_node.append_attribute("Name") = "types";
  type_node.append_attribute("format") = "ascii";
  int celltype = get_vtk_cell_type(topology.cell_type(), tdim);
  std::stringstream s;
  for (std::int32_t c = 0; c < num_cells; ++c)
    s << celltype << " ";
  type_node.append_child(pugi::node_pcdata).set_value(s.str().c_str());
}
//----------------------------------------------------------------------------
template <typename Scalar>
void write_function(
    const std::vector<std::reference_wrapper<const fem::Function<Scalar>>>& u,
    double time, std::unique_ptr<pugi::xml_document>& xml_doc,
    const std::string filename)
{
  if (!xml_doc)
    throw std::runtime_error("VTKFile has already been closed");

  if (u.empty())
    return;

  // Extract mesh
  assert(u.front().get().function_space());
  std::shared_ptr<const mesh::Mesh> mesh
      = u.front().get().function_space()->mesh();
  assert(mesh);
  for (std::size_t i = 1; i < u.size(); ++i)
  {
    if (u[i].get().function_space()->mesh() != mesh)
    {
      throw std::runtime_error(
          "Meshes for Functions to write to VTK file do not match.");
    }
  }
  // Get MPI comm
  const MPI_Comm comm = mesh->comm();
  const int mpi_rank = dolfinx::MPI::rank(comm);
  boost::filesystem::path p(filename);

  // Get the PVD "Collection" node
  pugi::xml_node xml_collections
      = xml_doc->child("VTKFile").child("Collection");
  assert(xml_collections);

  // Compute counter string
  const std::string counter_str = get_counter(xml_collections, "DataSet");

  // Get mesh data for this rank
  const mesh::Topology& topology = mesh->topology();
  const mesh::Geometry& geometry = mesh->geometry();
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
  add_mesh(*mesh, piece_node);

  // Loop through functions to add data types and ranks
  for (auto _u : u)
  {
    std::string data_type;
    if (is_cellwise(_u))
      data_type = "CellData";
    else
      data_type = "PointData";

    // Set last entry of a given rank to be the active data type
    // as Paraview only supports one active type
    if (piece_node.child(data_type.c_str()).empty())
      piece_node.append_child(data_type.c_str());
    const int rank = _u.get().function_space()->element()->value_shape().size();
    pugi::xml_node data_node = piece_node.child(data_type.c_str());
    std::string rank_type;
    if (rank == 0)
      rank_type = "Scalars";
    else if (rank == 1)
      rank_type = "Vectors";
    else if (rank == 2)
      rank_type = "Tensors";
    if (data_node.attribute(rank_type.c_str()).empty())
      data_node.append_attribute(rank_type.c_str());
    pugi::xml_attribute data = data_node.attribute(rank_type.c_str());
    data.set_value(_u.get().name.c_str());
  }

  // Add cell/point data to VTU node
  for (auto _u : u)
  {
    if (is_cellwise(_u))
    {
      const std::vector<Scalar> values
          = io::xdmf_utils::get_cell_data_values(_u);
      const size_t value_size
          = _u.get().function_space()->element()->value_size();
      assert(values.size() % value_size == 0);
      xt::xtensor<Scalar, 2> _values({values.size() / value_size, value_size});
      // FIXME: Avoid copies by writing directly a compound data
      for (std::size_t i = 0; i < _values.shape(0); ++i)
      {
        for (std::size_t j = 0; j < value_size; ++j)
        {
          _values(i, j) = values[i * value_size + j];
        }
      }
      pugi::xml_node data_node = piece_node.child("CellData");
      assert(!data_node.empty());
      add_data(_u, _values, data_node);
    }
    else
    {
      // Check if the function space is Lagrangian
      // NOTE: This should be changed if we add option to visualize DG
      std::shared_ptr<const dolfinx::fem::FiniteElement> element
          = _u.get().function_space()->element();
      if ((element->family().compare("Lagrange") == 0)
          or (element->family().compare("Q") == 0))
      {
        // Extract mesh data
        int tdim = mesh->topology().dim();
        auto cmap = mesh->geometry().cmap();
        const fem::ElementDofLayout geometry_layout = cmap.create_dof_layout();
        // Extract function value
        xtl::span<const Scalar> func_values = _u.get().x()->array();
        // Compute in tensor (one for scalar function, . . .)
        const size_t value_size_loc = element->value_size();

        // FIXME: Add proper interface for num coordinate dofs
        const graph::AdjacencyList<std::int32_t>& x_dofmap
            = mesh->geometry().dofmap();
        const int num_dofs_g = x_dofmap.num_links(0);

        auto map = mesh->topology().index_map(tdim);
        assert(map);
        const std::int32_t num_cells = map->size_local();

        // Resize array for holding point values
        xt::xtensor<Scalar, 2> point_values = xt::zeros<Scalar>(
            {mesh->geometry().x().size() / 3, value_size_loc});

        // If scalar function space
        if (element->num_sub_elements() == 0)
        {
          auto dofmap = _u.get().function_space()->dofmap();
          auto& element_layout = dofmap->element_dof_layout();
          for (std::int32_t i = 0; i <= tdim; i++)
          {
            // Check that subelement layout matches geometry layout
            if (geometry_layout.num_entity_dofs(i)
                != element_layout.num_entity_dofs(i))
            {
              LOG(WARNING) << "Output data is interpolated into a first order "
                              "Lagrange space.";
              point_values = _u.get().compute_point_values();
            }
          }
          // Loop through cells
          for (std::int32_t c = 0; c < num_cells; ++c)
          {
            // Get local to global dof ordering for geometry and function
            auto dofs = x_dofmap.links(c);
            auto cell_dofs = dofmap->cell_dofs(c);
            for (std::int32_t i = 0; i < num_dofs_g; i++)
            {
              point_values(dofs[i], 0) = func_values[cell_dofs[i]];
            }
          }
        }

        // Loop through each vector/tensor component
        bool element_matching_mesh = true;
        for (std::int32_t k = 0; k < element->num_sub_elements(); k++)
        {
          auto dofmap = _u.get().function_space()->sub({k})->dofmap();
          auto& element_layout = dofmap->element_dof_layout();

          for (std::int32_t i = 0; i <= tdim; i++)
          {
            // Check that subelement layout matches geometry layout
            if (geometry_layout.num_entity_dofs(i)
                != element_layout.num_entity_dofs(i))
            {
              element_matching_mesh = false;
              break;
              // throw std::runtime_error("Can only save Lagrange finite
              // element
              // "
              //                          "functions of same order "
              //                          "as the mesh geometry");
            }
          }
          if (element_matching_mesh)
          {
            // Loop through cells
            for (std::int32_t c = 0; c < num_cells; ++c)
            {
              // Get local to global dof ordering for geometry and function
              auto dofs = x_dofmap.links(c);
              auto cell_dofs = dofmap->cell_dofs(c);
              for (std::int32_t i = 0; i < num_dofs_g; i++)
              {
                point_values(dofs[i], k) = func_values[cell_dofs[i]];
              }
            }
          }
          else
          {
            LOG(WARNING) << "Output data is interpolated into a first order "
                            "Lagrange space.";
            break;
          }
        }
        if (!element_matching_mesh)
          point_values = _u.get().compute_point_values();
        pugi::xml_node data_node = piece_node.child("PointData");
        assert(!data_node.empty());
        add_data(_u, point_values, data_node);
      }
      else
      {
        LOG(WARNING) << "Output data is interpolated into a first order "
                        "Lagrange space.";
        xt::xtensor<Scalar, 2> point_values = _u.get().compute_point_values();
        pugi::xml_node data_node = piece_node.child("PointData");
        assert(!data_node.empty());
        add_data(_u, point_values, data_node);
        // throw std::runtime_error("Can only visualize Lagrange finite
        // elements");
      }
    }
  }

  // Save VTU XML to file
  boost::filesystem::path vtu(p.parent_path());
  if (!p.parent_path().empty())
    vtu += "/";
  vtu += p.stem().string() + "_p" + std::to_string(mpi_rank) + "_"
         + counter_str;
  vtu.replace_extension("vtu");
  xml_vtu.save_file(vtu.c_str(), "  ");

  // Create a PVTU XML object on rank 0
  boost::filesystem::path p_pvtu(p.parent_path());
  if (!p.parent_path().empty())
    p_pvtu += "/";
  p_pvtu += p.stem().string() + counter_str;
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
      if (is_cellwise(_u))
      {
        if (grid_node.child("PCellData").empty())
          grid_node.append_child("PCellData");
      }
      else
      {
        if (grid_node.child("PPointData").empty())
          grid_node.append_child("PPointData");
      }
    // Add mesh metadata to PVTU object
    add_pvtu_mesh(grid_node);
    // Add field data
    std::vector<std::string> components = {""};
    if constexpr (!std::is_scalar<Scalar>::value)
      components = {"real", "imag"};

    for (auto _u : u)
    {
      std::string d_type = is_cellwise(_u) ? "PCellData" : "PPointData";
      pugi::xml_node data_pnode = grid_node.child(d_type.c_str());
      const int rank
          = _u.get().function_space()->element()->value_shape().size();
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
          data_node.append_attribute("Name")
              = (component + "_" + _u.get().name).c_str();
        else
          data_node.append_attribute("Name")
              = (component + "" + _u.get().name).c_str();
        data_node.append_attribute("NumberOfComponents") = ncomps;
      }

      // Add data for each process to the PVTU object
      const int mpi_size = dolfinx::MPI::size(comm);
      for (int i = 0; i < mpi_size; ++i)
      {
        boost::filesystem::path vtu = p.stem();
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
io::VTKFile::VTKFile(MPI_Comm comm, const std::string filename,
                     const std::string)
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
void io::VTKFile::write(
    const std::vector<std::reference_wrapper<const fem::Function<double>>>& u,
    double time)
{
  write_function(u, time, _pvd_xml, _filename);
}
//----------------------------------------------------------------------------
void io::VTKFile::write(
    const std::vector<
        std::reference_wrapper<const fem::Function<std::complex<double>>>>& u,
    double time)
{
  write_function(u, time, _pvd_xml, _filename);
}
//----------------------------------------------------------------------------
void io::VTKFile::write(const mesh::Mesh& mesh, double time)
{
  if (!_pvd_xml)
    throw std::runtime_error("VTKFile has already been closed");

  const int mpi_rank = MPI::rank(_comm.comm());
  boost::filesystem::path p(_filename);

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
  add_mesh(mesh, piece_node);

  // Save VTU XML to file
  boost::filesystem::path vtu(p.parent_path());
  if (!p.parent_path().empty())
    vtu += "/";
  vtu += p.stem().string() + "_p" + std::to_string(mpi_rank) + "_"
         + counter_str;
  vtu.replace_extension("vtu");
  xml_vtu.save_file(vtu.c_str(), "  ");

  // Create a PVTU XML object on rank 0
  boost::filesystem::path p_pvtu(p.parent_path());
  if (!p.parent_path().empty())
    p_pvtu += "/";
  p_pvtu += p.stem().string() + counter_str;
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
      boost::filesystem::path vtu = p.stem();
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
