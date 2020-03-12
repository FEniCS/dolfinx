// Copyright (C) 2012-2016 Chris N. Richardson and Garth N. Wells
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
void xdmf_function::write(const function::Function& u, double t, int counter,
                          pugi::xml_document& xml_doc, hid_t h5_id)
{
  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);

  pugi::xml_node xdmf_node = xml_doc.child("Xdmf");
  assert(xdmf_node);
  pugi::xml_node domain_node = xdmf_node.child("Domain");
  assert(domain_node);

  // Should functions share mesh or not? By default they do not
  std::string tg_name = std::string("TimeSeries_") + u.name;

  // Look for existing time series grid node with Name == tg_name
  bool new_timegrid = false;
  std::string time_step_str = boost::lexical_cast<std::string>(t);
  pugi::xml_node timegrid_node, mesh_node;
  timegrid_node
      = domain_node.find_child_by_attribute("Grid", "Name", tg_name.c_str());

  // Ensure that we have a time series grid node
  if (timegrid_node)
  {
    // Get existing mesh grid node with the correct time step if it exist
    // (otherwise null)
    std::string xpath = std::string("Grid[Time/@Value=\"") + time_step_str
                        + std::string("\"]");
    mesh_node = timegrid_node.select_node(xpath.c_str()).node();
    assert(std::string(timegrid_node.attribute("CollectionType").value())
           == "Temporal");
  }
  else
  {
    //  Create a new time series grid node with Name = tg_name
    timegrid_node = domain_node.append_child("Grid");
    assert(timegrid_node);
    timegrid_node.append_attribute("Name") = tg_name.c_str();
    timegrid_node.append_attribute("GridType") = "Collection";
    timegrid_node.append_attribute("CollectionType") = "Temporal";
    new_timegrid = true;
  }

  // Only add mesh grid node at this time step if no other function has
  // previously added it (and functions_share_mesh == true)
  if (!mesh_node)
  {
    // Add the mesh grid node to to the time series grid node
    // if (new_timegrid or rewrite_function_mesh)
    if (new_timegrid)
    {
      xdmf_mesh::add_mesh(mesh->mpi_comm(), timegrid_node, h5_id, *mesh,
                          "/Mesh/" + std::to_string(counter));
    }
    else
    {
      // Make a grid node that references back to first mesh grid node of the
      // time series
      pugi::xml_node grid_node = timegrid_node.append_child("Grid");
      assert(grid_node);

      // Reference to previous topology and geometry document nodes via
      // XInclude
      std::string xpointer
          = std::string("xpointer(//Grid[@Name=\"") + tg_name
            + std::string("\"]/Grid[1]/*[self::Topology or self::Geometry])");
      pugi::xml_node reference = grid_node.append_child("xi:include");
      assert(reference);
      reference.append_attribute("xpointer") = xpointer.c_str();
    }

    // Get the newly created mesh grid node
    mesh_node = timegrid_node.last_child();
    assert(mesh_node);

    // Add time value to mesh grid node
    pugi::xml_node time_node = mesh_node.append_child("Time");
    time_node.append_attribute("Value") = time_step_str.c_str();
  }

  LOG(INFO) << "Adding function to node \"" << mesh_node.path('/') << "\"";

  std::string element_family = u.function_space()->element()->family();
  const bool use_mpi_io = (dolfinx::MPI::size(mesh->mpi_comm()) > 1);
  const std::size_t element_degree = u.function_space()->element()->degree();
  const mesh::CellType element_cell_type
      = u.function_space()->element()->cell_shape();

  // Map of standard UFL family abbreviations for visualisation
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
      = {{mesh::CellType::interval, "interval"},
         {mesh::CellType::triangle, "triangle"},
         {mesh::CellType::tetrahedron, "tetrahedron"},
         {mesh::CellType::quadrilateral, "quadrilateral"},
         {mesh::CellType::hexahedron, "hexahedron"}};

  // Check that element is supported
  auto const it = family_abbr.find(element_family);
  if (it == family_abbr.end())
    throw std::runtime_error("Element type not supported for XDMF output.");
  element_family = it->second;

  // Check that cell shape is supported
  auto it_shape = cell_shape_repr.find(element_cell_type);
  if (it_shape == cell_shape_repr.end())
    throw std::runtime_error("Cell type not supported for XDMF output.");
  const std::string element_cell = it_shape->second;

  // Prepare main Attribute for the FiniteElementFunction type

  std::string function_time_name = u.name + "_" + std::to_string(counter);
  std::string h5_path = u.name + "/" + function_time_name;
  std::string attr_name;
#ifdef PETSC_USE_COMPLEX
  std::vector<std::string> components = {"real", "imag"};
#else
  std::vector<std::string> components = {""};
#endif

  // Write function u (for each of its components)
  for (const std::string component : components)
  {
    if (component.empty())
      attr_name = u.name;
    else
    {
      attr_name = component + "_" + u.name;
      h5_path = h5_path + "/" + component;
    }

    pugi::xml_node fe_attribute_node = mesh_node.append_child("Attribute");
    fe_attribute_node.append_attribute("ItemType") = "FiniteElementFunction";
    fe_attribute_node.append_attribute("ElementFamily")
        = element_family.c_str();
    fe_attribute_node.append_attribute("ElementDegree")
        = std::to_string(element_degree).c_str();
    fe_attribute_node.append_attribute("ElementCell") = element_cell.c_str();
    fe_attribute_node.append_attribute("Name") = attr_name.c_str();
    fe_attribute_node.append_attribute("Center") = "Other";
    fe_attribute_node.append_attribute("AttributeType")
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

    const std::int64_t num_cell_dofs_global
        = MPI::sum(mesh->mpi_comm(), cell_dofs.size());

    // Write dofmap = indices to the values DataItem
    xdmf_utils::add_data_item(fe_attribute_node, h5_id, h5_path + "/cell_dofs",
                              cell_dofs, offset, {num_cell_dofs_global, 1},
                              "Int", use_mpi_io);

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

#ifdef PETSC_USE_COMPLEX
    // FIXME: Avoid copies by writing directly a compound data
    std::vector<double> component_data_values(local_data.size());
    for (std::size_t i = 0; i < local_data.size(); i++)
    {
      if (component == "real")
        component_data_values[i] = local_data[i].real();
      else if (component == "imag")
        component_data_values[i] = local_data[i].imag();
    }
    const std::size_t offset = MPI::global_offset(
        mesh->mpi_comm(), component_data_values.size(), true);
    xdmf_utils::add_data_item(
        fe_attribute_node, h5_id, h5_path + "/vector", component_data_values,
        offset{(std::int64_t)u_vector.size(), 1}, "Float", use_mpi_io);
#else
    const std::size_t offset_data
        = MPI::global_offset(mesh->mpi_comm(), local_data.size(), true);
    xdmf_utils::add_data_item(
        fe_attribute_node, h5_id, h5_path + "/vector", local_data, offset_data,
        {(std::int64_t)u_vector.size(), 1}, "Float", use_mpi_io);
#endif
  }
}
