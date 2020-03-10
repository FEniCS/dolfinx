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

/// Returns true for DG0 function::Functions
bool has_cell_centred_data(const function::Function& u)
{
  int cell_based_dim = 1;
  for (int i = 0; i < u.value_rank(); i++)
    cell_based_dim *= u.function_space()->mesh()->topology().dim();

  assert(u.function_space());
  assert(u.function_space()->dofmap());
  assert(u.function_space()->dofmap()->element_dof_layout);
  return (u.function_space()->dofmap()->element_dof_layout->num_dofs()
          == cell_based_dim);
}
//-----------------------------------------------------------------------------

// Get data width - normally the same as u.value_size(), but expand for
// 2D vector/tensor because XDMF presents everything as 3D
int get_padded_width(const function::Function& u)
{
  const int width = u.value_size();
  const int rank = u.value_rank();
  if (rank == 1 and width == 2)
    return 3;
  else if (rank == 2 and width == 4)
    return 9;
  return width;
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
  // if (functions_share_mesh)
  //   tg_name = "TimeSeries";

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

  // Get function::Function data values and shape
  std::vector<PetscScalar> data_values;
  const bool cell_centred = has_cell_centred_data(u);
  if (cell_centred)
    data_values = xdmf_utils::get_cell_data_values(u);
  else
    data_values = xdmf_utils::get_point_data_values(u);

  auto map_c = mesh->topology().index_map(mesh->topology().dim());
  assert(map_c);

  // FIXME: Should this be the geometry map?
  auto map_v = mesh->topology().index_map(0);
  assert(map_v);

  // Add attribute DataItem node and write data
  const int width = get_padded_width(u);
  assert(data_values.size() % width == 0);
  const int num_values
      = cell_centred ? map_c->size_global() : map_v->size_global();

#ifdef PETSC_USE_COMPLEX
  std::vector<std::string> components = {"real", "imag"};
#else
  std::vector<std::string> components = {""};
#endif

  for (const std::string component : components)
  {
    std::string attr_name;
    std::string dataset_name;
    if (component.empty())
    {
      attr_name = u.name;
      dataset_name = "/VisualisationVector/" + std::to_string(counter);
    }
    else
    {
      attr_name = component + "_" + u.name;
      dataset_name
          = "/VisualisationVector/" + component + "/" + std::to_string(counter);
    }
    // Add attribute node
    pugi::xml_node attribute_node = mesh_node.append_child("Attribute");
    assert(attribute_node);
    attribute_node.append_attribute("Name") = attr_name.c_str();
    attribute_node.append_attribute("AttributeType")
        = rank_to_string(u.value_rank()).c_str();
    attribute_node.append_attribute("Center") = cell_centred ? "Cell" : "Node";

    const bool use_mpi_io = (dolfinx::MPI::size(mesh->mpi_comm()) > 1);
#ifdef PETSC_USE_COMPLEX
    // FIXME: Avoid copies by writing directly a compound data
    std::vector<double> component_data_values(data_values.size());
    for (std::size_t i = 0; i < data_values.size(); i++)
    {
      if (component == components[0])
        component_data_values[i] = data_values[i].real();
      else if (component == components[1])
        component_data_values[i] = data_values[i].imag();
    }

    // Add data item of component
    const std::int64_t offset = dolfinx::MPI::global_offset(
        mesh->mpi_comm(), component_data_values.size() / width, true);
    xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name,
                              component_data_values, offset,
                              {num_values, width}, "", use_mpi_io);
#else
    // Add data item
    const std::int64_t offset = dolfinx::MPI::global_offset(
        mesh->mpi_comm(), data_values.size() / width, true);
    xdmf_utils::add_data_item(attribute_node, h5_id, dataset_name, data_values,
                              offset, {num_values, width}, "", use_mpi_io);
#endif
  }
}
//-----------------------------------------------------------------------------
