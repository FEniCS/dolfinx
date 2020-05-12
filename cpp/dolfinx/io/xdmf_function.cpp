// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "xdmf_function.h"
#include "pugixml.hpp"
#include "xdmf_mesh.h"
#include "xdmf_utils.h"
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <string>

using namespace dolfinx;
using namespace dolfinx::io;

namespace
{
//-----------------------------------------------------------------------------

// Convert a value_rank to the XDMF string description (Scalar, Vector,
// Tensor)
std::string rank_to_string(int value_rank)
{
  switch (value_rank)
  {
  case 0:
    return "Scalar";
  case 1:
    return "Vector";
  case 2:
    return "Tensor";
  default:
    throw std::runtime_error("Range Error");
  }
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
void xdmf_function::add_function(MPI_Comm comm, const function::Function& u,
                                 const double t, pugi::xml_node& xml_node,
                                 const hid_t h5_id)
{
  LOG(INFO) << "Adding function to node \"" << xml_node.path('/') << "\"";

  assert(u.function_space());
  std::shared_ptr<const mesh::Mesh> mesh = u.function_space()->mesh();
  assert(mesh);
  std::string element_family = u.function_space()->element()->family();
  const bool use_mpi_io = (dolfinx::MPI::size(mesh->mpi_comm()) > 1);
  const std::size_t element_degree = u.function_space()->element()->degree();
  const mesh::CellType element_cell_type
      = u.function_space()->element()->cell_shape();

  // Map of standard UFL family abbreviations for visualization
  const std::map<std::string, std::string> family_abbr
      = {{"Lagrange", "CG"},
         {"Discontinuous Lagrange", "DG"},
         {"Raviart-Thomas", "RT"},
         {"Brezzi-Douglas-Marini", "BDM"},
         {"Crouzeix-Raviart", "CR"},
         {"Nedelec 1st kind H(curl)", "N1curl"},
         {"Nedelec 2nd kind H(curl)", "N2curl"},
         {"Q", "Q"},
         {"DQ", "DQ"}};

  const std::map<mesh::CellType, std::string> cell_shape_repr
      = {{dolfinx::mesh::CellType::interval, "interval"},
         {dolfinx::mesh::CellType::triangle, "triangle"},
         {dolfinx::mesh::CellType::tetrahedron, "tetrahedron"},
         {dolfinx::mesh::CellType::quadrilateral, "quadrilateral"},
         {dolfinx::mesh::CellType::hexahedron, "hexahedron"}};
  // Check that element is supported
  auto const it = family_abbr.find(element_family);
  if (it == family_abbr.end())
    throw std::runtime_error(
        "Element type not supported for XDMF output. Please project function "
        "to a suitable finite element space");
  element_family = it->second;

  // Check that cell shape is supported
  auto it_shape = cell_shape_repr.find(element_cell_type);
  if (it_shape == cell_shape_repr.end())
    throw std::runtime_error("Cell type not supported for XDMF output.");

  const std::string element_cell = it_shape->second;

  auto map_c = mesh->topology().index_map(mesh->topology().dim());
  assert(map_c);

#ifdef PETSC_USE_COMPLEX
  const std::vector<std::string> components = {"real", "imag"};
#else
  const std::vector<std::string> components = {""};
#endif

  std::string t_str = boost::lexical_cast<std::string>(t);
  std::replace(t_str.begin(), t_str.end(), '.', '_');

  for (const auto& component : components)
  {
    std::string attr_name;
    std::string dataset_name;
    if (component.empty())
    {
      attr_name = u.name;
      dataset_name = "/Function/" + attr_name + "/" + t_str;
    }
    else
    {
      attr_name = component + "_" + u.name;
      dataset_name = "/Function/" + attr_name + "/" + t_str;
    }
    // Add attribute node
    pugi::xml_node attribute_node = xml_node.append_child("Attribute");
    assert(attribute_node);
    attribute_node.append_attribute("ItemType") = "FiniteElementFunction";
    attribute_node.append_attribute("ElementFamily") = element_family.c_str();
    attribute_node.append_attribute("ElementDegree")
        = std::to_string(element_degree).c_str();
    attribute_node.append_attribute("ElementCell") = element_cell.c_str();
    attribute_node.append_attribute("Name") = attr_name.c_str();
    attribute_node.append_attribute("Center") = "Other";
    attribute_node.append_attribute("AttributeType")
        = rank_to_string(u.value_rank()).c_str();

    // Prepare and save number of dofs per cell (x_cell_dofs) and cell
    // dofmaps (cell_dofs)

    assert(u.function_space()->dofmap());
    const fem::DofMap& dofmap = *u.function_space()->dofmap();

    const std::size_t tdim = mesh->topology().dim();
    std::vector<std::int32_t> cell_dofs;
    std::vector<std::size_t> x_cell_dofs;
    const std::size_t n_cells = mesh->topology().index_map(tdim)->size_local();
    x_cell_dofs.reserve(n_cells);

    Eigen::Array<std::int64_t, Eigen::Dynamic, 1> local_to_global_map
        = dofmap.index_map->indices(true);

    // Add number of dofs for each cell
    // Add cell dofmap
    for (std::size_t i = 0; i != n_cells; ++i)
    {
      x_cell_dofs.push_back(cell_dofs.size());
      auto cell_dofs_i = dofmap.cell_dofs(i);
      for (Eigen::Index j = 0; j < cell_dofs_i.size(); ++j)
      {
        auto p = cell_dofs_i[j];
        assert(p < (std::int32_t)local_to_global_map.size());
        cell_dofs.push_back(local_to_global_map[p]);
      }
    }

    // Add offset to CSR index to be seamless in parallel
    const std::size_t offset
        = MPI::global_offset(mesh->mpi_comm(), cell_dofs.size(), true);
    for (auto& x : x_cell_dofs)
      x += offset;

    std::int64_t num_cell_dofs_global = 0;
    size_t local_size = cell_dofs.size();
    MPI_Allreduce(&local_size, &num_cell_dofs_global, 1, MPI_INT64_T, MPI_SUM,
                  comm);

    // Write dofmap = indices to the values DataItem
    xdmf_utils::add_data_item(attribute_node, h5_id,
                              dataset_name + "/cell_dofs", cell_dofs, offset,
                              {num_cell_dofs_global, 1}, "Int", use_mpi_io);

    // FIXME: Avoid unnecessary copying of data
    // Get all local data
    const la::PETScVector& u_vector = u.vector();
    PetscErrorCode ierr;
    const PetscScalar* u_ptr = nullptr;
    ierr = VecGetArrayRead(u_vector.vec(), &u_ptr);
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "VecGetArrayRead");
    std::vector<PetscScalar> local_data(u_ptr, u_ptr + u_vector.local_size());
    ierr = VecRestoreArrayRead(u_vector.vec(), &u_ptr);
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "VecRestoreArrayRead");

    const bool use_mpi_io = (dolfinx::MPI::size(comm) > 1);
#ifdef PETSC_USE_COMPLEX
    // FIXME: Avoid copies by writing directly a compound data
    std::vector<double> component_local_data(local_data.size());
    if (component == "real")
    {
      for (std::size_t i = 0; i < local_data.size(); i++)
        component_local_data[i] = local_data[i].real();
    }
    else if (component == "imag")
    {
      for (std::size_t i = 0; i < local_data.size(); i++)
        component_local_data[i] = local_data[i].imag();
    }

    // Add data item of component
    const std::int64_t offset_data
        = dolfinx::MPI::global_offset(comm, component_local_data.size(), true);
    xdmf_utils::add_data_item(
        attribute_node, h5_id, dataset_name, component_local_data, offset_data,
        {(std::int64_t)u_vector.size(), 1}, "Float", use_mpi_io);
#else
    // Add data item
    const std::int64_t offset_real
        = dolfinx::MPI::global_offset(comm, local_data.size(), true);
    xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name, local_data,
                              offset_real, {(std::int64_t)u_vector.size(), 1},
                              "Float", use_mpi_io);
#endif
  }
}
//-----------------------------------------------------------------------------
