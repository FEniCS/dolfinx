// Copyright (C) 2012-2020 Chris N. Richardson, Garth N. Wells and Michal Habera
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "XDMFFile.h"
#include "cells.h"
#include "xdmf_function.h"
#include "xdmf_mesh.h"
#include "xdmf_utils.h"
#include <boost/lexical_cast.hpp>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>
#include <filesystem>
#include <pugixml.hpp>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
XDMFFile::XDMFFile(MPI_Comm comm, const std::filesystem::path& filename,
                   std::string file_mode, Encoding encoding)
    : _comm(comm), _filename(filename), _file_mode(file_mode),
      _xml_doc(new pugi::xml_document), _encoding(encoding)
{
  // Handle HDF5 and XDMF files with the file mode. At the end of this
  // we will have _hdf5_file and _xml_doc both pointing to a valid and
  // opened file handles.

  if (_encoding == Encoding::HDF5)
  {
    // See https://www.hdfgroup.org/hdf5-quest.html#gzero on zero for
    // _hdf5_file_id(0)

    // Open HDF5 file
    const std::filesystem::path hdf5_filename
        = xdmf_utils::get_hdf5_filename(_filename);
    const bool mpi_io = dolfinx::MPI::size(_comm.comm()) > 1 ? true : false;
    _h5_id
        = io::hdf5::open_file(_comm.comm(), hdf5_filename, file_mode, mpi_io);
    assert(_h5_id > 0);
    spdlog::info("Opened HDF5 file with id \"{}\"", _h5_id);
  }
  else
  {
    // HDF handle be -1 to avoid closing a HDF file on destruction
    _h5_id = -1;
  }

  if (_file_mode == "r")
  {
    // Load XML doc from file
    pugi::xml_parse_result result = _xml_doc->load_file(_filename.c_str());
    if (!result)
      throw std::runtime_error("Failed to load xml document from file.");

    if (_xml_doc->child("Xdmf").empty())
      throw std::runtime_error("Empty <Xdmf> root node.");

    if (_xml_doc->child("Xdmf").child("Domain").empty())
      throw std::runtime_error("Empty <Domain> node.");
  }
  else if (_file_mode == "w")
  {
    if (_encoding == Encoding::ASCII and dolfinx::MPI::size(_comm.comm()) > 1)
    {
      throw std::runtime_error(
          "ASCII encoding is not supported for writing files in parallel.");
    }

    _xml_doc->reset();

    // Add XDMF node and version attribute
    _xml_doc->append_child(pugi::node_doctype)
        .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
    pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
    assert(xdmf_node);
    xdmf_node.append_attribute("Version") = "3.0";
    xdmf_node.append_attribute("xmlns:xi") = "https://www.w3.org/2001/XInclude";

    pugi::xml_node domain_node = xdmf_node.append_child("Domain");
    if (!domain_node)
      throw std::runtime_error("Failed to append xml/xdmf Domain.");
  }
  else if (_file_mode == "a")
  {
    if (_encoding == Encoding::ASCII and dolfinx::MPI::size(_comm.comm()) > 1)
    {
      throw std::runtime_error("ASCII encoding is not supported for appending "
                               "to files in parallel.");
    }

    if (std::filesystem::exists(_filename))
    {
      // Load XML doc from file
      [[maybe_unused]] pugi::xml_parse_result result
          = _xml_doc->load_file(_filename.c_str());
      assert(result);

      if (_xml_doc->child("Xdmf").empty())
        throw std::runtime_error("Empty <Xdmf> root node.");

      if (_xml_doc->child("Xdmf").child("Domain").empty())
        throw std::runtime_error("Empty <Domain> node.");
    }
    else
    {
      _xml_doc->reset();

      // Add XDMF node and version attribute
      _xml_doc->append_child(pugi::node_doctype)
          .set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
      pugi::xml_node xdmf_node = _xml_doc->append_child("Xdmf");
      assert(xdmf_node);
      xdmf_node.append_attribute("Version") = "3.0";
      xdmf_node.append_attribute("xmlns:xi")
          = "https://www.w3.org/2001/XInclude";

      pugi::xml_node domain_node = xdmf_node.append_child("Domain");
      if (!domain_node)
        throw std::runtime_error("Failed to append xml/xdmf Domain.");
    }
  }
}
//-----------------------------------------------------------------------------
XDMFFile::~XDMFFile() { close(); }
//-----------------------------------------------------------------------------
void XDMFFile::close()
{
  if (_h5_id > 0)
    io::hdf5::close_file(_h5_id);
  _h5_id = -1;
}
//-----------------------------------------------------------------------------
template <std::floating_point U>
void XDMFFile::write_mesh(const mesh::Mesh<U>& mesh, std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  // Add the mesh Grid to the domain
  xdmf_mesh::add_mesh(_comm.comm(), node, _h5_id, mesh, mesh.name);

  // Save XML file (on process 0 only)
  if (MPI::rank(_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
/// @cond
template void XDMFFile::write_mesh(const mesh::Mesh<double>&, std::string);
template void XDMFFile::write_mesh(const mesh::Mesh<float>&, std::string);
/// @endcond
//-----------------------------------------------------------------------------
void XDMFFile::write_geometry(const mesh::Geometry<double>& geometry,
                              std::string name, std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  // Prepare a Grid for Geometry only
  pugi::xml_node grid_node = node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  const std::string path_prefix = "/Geometry/" + name;
  xdmf_mesh::add_geometry_data(_comm.comm(), grid_node, _h5_id, path_prefix,
                               geometry);

  // Save XML file (on process 0 only)
  if (MPI::rank(_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
mesh::Mesh<double>
XDMFFile::read_mesh(const fem::CoordinateElement<double>& element,
                    mesh::GhostMode mode, std::string name,
                    std::string xpath) const
{
  // Read mesh data
  auto [cells, cshape] = XDMFFile::read_topology_data(name, xpath);
  auto [x, xshape] = XDMFFile::read_geometry_data(name, xpath);

  // Create mesh
  const std::vector<double>& _x = std::get<std::vector<double>>(x);
  mesh::Mesh<double> mesh
      = mesh::create_mesh(_comm.comm(), cells, element, _x, xshape, mode);
  mesh.name = name;
  return mesh;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<std::int64_t>, std::array<std::size_t, 2>>
XDMFFile::read_topology_data(std::string name, std::string xpath) const
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  pugi::xml_node grid_node
      = node.select_node(("Grid[@Name='" + name + "']").c_str()).node();
  if (!grid_node)
    throw std::runtime_error("<Grid> with name '" + name + "' not found.");

  spdlog::info("Read topology data \"{}\" at {}", name, xpath);
  return xdmf_mesh::read_topology_data(_comm.comm(), _h5_id, grid_node);
}
//-----------------------------------------------------------------------------
std::pair<std::variant<std::vector<float>, std::vector<double>>,
          std::array<std::size_t, 2>>
XDMFFile::read_geometry_data(std::string name, std::string xpath) const
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  pugi::xml_node grid_node
      = node.select_node(("Grid[@Name='" + name + "']").c_str()).node();
  if (!grid_node)
    throw std::runtime_error("<Grid> with name '" + name + "' not found.");

  spdlog::info("Read geometry data \"{}\" at {}", name, xpath);
  return xdmf_mesh::read_geometry_data(_comm.comm(), _h5_id, grid_node);
}
//-----------------------------------------------------------------------------
template <dolfinx::scalar T, std::floating_point U>
void XDMFFile::write_function(const fem::Function<T, U>& u, double t,
                              std::string mesh_xpath)
{
  assert(_xml_doc);

  std::string timegrid_xpath
      = "/Xdmf/Domain/Grid[@GridType='Collection'][@Name='" + u.name + "']";
  pugi::xml_node timegrid_node
      = _xml_doc->select_node(timegrid_xpath.c_str()).node();

  if (!timegrid_node)
  {
    pugi::xml_node domain_node = _xml_doc->select_node("/Xdmf/Domain").node();
    timegrid_node = domain_node.append_child("Grid");
    timegrid_node.append_attribute("Name") = u.name.c_str();
    timegrid_node.append_attribute("GridType") = "Collection";
    timegrid_node.append_attribute("CollectionType") = "Temporal";
  }

  assert(timegrid_node);

  pugi::xml_node grid_node = timegrid_node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = u.name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  pugi::xml_node mesh_node = _xml_doc->select_node(mesh_xpath.c_str()).node();
  if (!mesh_node)
  {
    spdlog::warn("No mesh found at '{}'. Write mesh before function!",
                 mesh_xpath);
  }

  const std::string ref_path
      = "xpointer(" + mesh_xpath + "/*[self::Topology or self::Geometry])";

  pugi::xml_node topo_geo_ref = grid_node.append_child("xi:include");
  topo_geo_ref.append_attribute("xpointer") = ref_path.c_str();
  assert(topo_geo_ref);

  std::string t_str = boost::lexical_cast<std::string>(t);
  pugi::xml_node time_node = grid_node.append_child("Time");
  time_node.append_attribute("Value") = t_str.c_str();
  assert(time_node);

  // Add the mesh Grid to the domain
  xdmf_function::add_function(_comm.comm(), u, t, grid_node, _h5_id);

  // Save XML file (on process 0 only)
  if (dolfinx::MPI::rank(_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
// Instantiation for different types
/// @cond
template void XDMFFile::write_function(const fem::Function<float, float>&,
                                       double, std::string);
template void XDMFFile::write_function(const fem::Function<double, double>&,
                                       double, std::string);
template void
XDMFFile::write_function(const fem::Function<std::complex<float>, float>&,
                         double, std::string);
template void
XDMFFile::write_function(const fem::Function<std::complex<double>, double>&,
                         double, std::string);
/// @endcond
//-----------------------------------------------------------------------------
template <typename U, std::floating_point T>
void XDMFFile::write_meshtags(const mesh::MeshTags<U>& meshtags,
                              const mesh::Geometry<T>& x,
                              std::string geometry_xpath, std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  pugi::xml_node grid_node = node.append_child("Grid");
  assert(grid_node);
  grid_node.append_attribute("Name") = meshtags.name.c_str();
  grid_node.append_attribute("GridType") = "Uniform";

  const std::string geo_ref_path = "xpointer(" + geometry_xpath + ")";
  pugi::xml_node geo_ref_node = grid_node.append_child("xi:include");
  geo_ref_node.append_attribute("xpointer") = geo_ref_path.c_str();
  assert(geo_ref_node);
  xdmf_mesh::add_meshtags(_comm.comm(), meshtags, x, grid_node, _h5_id,
                          meshtags.name);

  // Save XML file (on process 0 only)
  if (MPI::rank(_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
// Instantiation for different types
/// @cond
template void XDMFFile::write_meshtags(const mesh::MeshTags<std::int32_t>&,
                                       const mesh::Geometry<float>& x,
                                       std::string, std::string);
template void XDMFFile::write_meshtags(const mesh::MeshTags<std::int32_t>&,
                                       const mesh::Geometry<double>& x,
                                       std::string, std::string);
template void XDMFFile::write_meshtags(const mesh::MeshTags<std::int64_t>&,
                                       const mesh::Geometry<float>& x,
                                       std::string, std::string);
template void XDMFFile::write_meshtags(const mesh::MeshTags<std::int64_t>&,
                                       const mesh::Geometry<double>& x,
                                       std::string, std::string);
template void XDMFFile::write_meshtags(const mesh::MeshTags<float>&,
                                       const mesh::Geometry<float>& x,
                                       std::string, std::string);
template void XDMFFile::write_meshtags(const mesh::MeshTags<float>&,
                                       const mesh::Geometry<double>& x,
                                       std::string, std::string);
template void XDMFFile::write_meshtags(const mesh::MeshTags<double>&,
                                       const mesh::Geometry<float>& x,
                                       std::string, std::string);
template void XDMFFile::write_meshtags(const mesh::MeshTags<double>&,
                                       const mesh::Geometry<double>& x,
                                       std::string, std::string);
/// @endcond
//-----------------------------------------------------------------------------
template <typename T>
mesh::MeshTags<T>
XDMFFile::read_meshtags(const mesh::Mesh<double>& mesh, std::string name,
                        std::optional<std::string> attribute_name,
                        std::string xpath)
{
  spdlog::info("XDMF read meshtags ({})", name);
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");
  pugi::xml_node grid_node
      = node.select_node(("Grid[@Name='" + name + "']").c_str()).node();
  if (!grid_node)
    throw std::runtime_error("<Grid> with name '" + name + "' not found.");

  const auto [entities, eshape] = read_topology_data(name, xpath);

  pugi::xml_node values_data_node
      = grid_node.child("Attribute").child("DataItem");
  if (attribute_name)
  {
    // Search for a child that contains an attribute with the requested name
    pugi::xml_node attribute_node = grid_node.find_child(
        [attr_name = *attribute_name](auto n)
        { return n.attribute("Name").value() == attr_name; });
    if (!attribute_node)
    {
      throw std::runtime_error("Attribute with name '" + *attribute_name
                               + "' not found.");
    }
    else
    {
      values_data_node = attribute_node.child("DataItem");
    }
  }

  // Now we need to figure out what type of data the file contains and check
  // that it is compatible with what we are reading.

  // If the data is stored in HDF format
  if (const auto& format_attribute = values_data_node.attribute("Format");
      format_attribute && std::string(format_attribute.value()) == "HDF")
  {
    const auto& [file_name, node_path]
        = xdmf_utils::get_hdf5_paths(values_data_node);

    // Then we need to read the data type from the .h5 file
    const auto datatype_id
        = io::hdf5::get_dataset_datatype(_h5_id, node_path.c_str());

    if (!H5Tequal(datatype_id, io::hdf5::hdf5_type<T>()))
    {
      throw std::runtime_error(
          "The data in the XDMF file does not match the required data type.");
    }
  }
  else
  {
    // Otherwise, this information should be stored in node attributes in the
    // XML file
    std::string data_type;

    // "DataType" is the attribute used in the current version of the
    // standard...
    if (const auto& data_type_attribute
        = values_data_node.attribute("DataType");
        data_type_attribute)
    {
      data_type = data_type_attribute.value();
    }
    // ... But "NumberType" is supported too and widely used in its place
    else if (const auto& data_type_attribute
             = values_data_node.attribute("NumberType");
             data_type_attribute)
    {
      data_type = data_type_attribute.value();
    }
    // If unspecified, use the standard's implicit default
    else
    {
      data_type = "Float";
    }

    int precision;

    // If there is an attribute that specifies the precision, read it from there
    if (const auto& precision_attribute
        = values_data_node.attribute("Precision");
        precision_attribute)
    {
      precision = std::stol(values_data_node.attribute("Precision").value());
    }
    // otherwhise use the standard's default
    else
    {
      precision = 4;
    }

    if (data_type != xdmf_integral_float<T>::data_type
        || precision != xdmf_integral_float<T>::precision)
    {
      throw std::runtime_error(
          "The data in the XDMF file does not match the required data type.");
    }
  }

  const std::vector values
      = xdmf_utils::get_dataset<T>(_comm.comm(), values_data_node, _h5_id);

  const std::pair<std::string, int> cell_type_str
      = xdmf_utils::get_cell_type(grid_node.child("Topology"));
  mesh::CellType cell_type = mesh::to_type(cell_type_str.first);

  // Permute entities from VTK to DOLFINx ordering
  std::vector<std::int64_t> entities1 = io::cells::apply_permutation(
      entities, eshape, io::cells::perm_vtk(cell_type, eshape[1]));

  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int64_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      entities_span(entities1.data(), eshape);
  std::pair<std::vector<std::int32_t>, std::vector<T>> entities_values
      = xdmf_utils::distribute_entity_data<T>(
          *mesh.topology(), mesh.geometry().input_global_indices(),
          mesh.geometry().index_map()->size_global(),
          mesh.geometry().cmap().create_dof_layout(), mesh.geometry().dofmap(),
          mesh::cell_dim(cell_type), entities_span, values);

  spdlog::info("XDMF create meshtags");
  std::size_t num_vertices_per_entity = mesh::cell_num_entities(
      mesh::cell_entity_type(mesh.topology()->cell_type(),
                             mesh::cell_dim(cell_type), 0),
      0);
  const graph::AdjacencyList<std::int32_t> entities_adj
      = graph::regular_adjacency_list(std::move(entities_values.first),
                                      num_vertices_per_entity);
  mesh::MeshTags meshtags = mesh::create_meshtags(
      mesh.topology(), mesh::cell_dim(cell_type), entities_adj,
      std::span<const T>(entities_values.second));
  meshtags.name = name;

  return meshtags;
}
//-----------------------------------------------------------------------------
// Instantiation for different types
/// @cond
template mesh::MeshTags<std::int32_t>
XDMFFile::read_meshtags(const mesh::Mesh<double>& mesh, std::string name,
                        std::optional<std::string> attribute_name,
                        std::string xpath);
template mesh::MeshTags<std::int64_t>
XDMFFile::read_meshtags(const mesh::Mesh<double>& mesh, std::string name,
                        std::optional<std::string> attribute_name,
                        std::string xpath);
template mesh::MeshTags<float>
XDMFFile::read_meshtags(const mesh::Mesh<double>& mesh, std::string name,
                        std::optional<std::string> attribute_name,
                        std::string xpath);
template mesh::MeshTags<double>
XDMFFile::read_meshtags(const mesh::Mesh<double>& mesh, std::string name,
                        std::optional<std::string> attribute_name,
                        std::string xpath);
/// @endcond
//-----------------------------------------------------------------------------
std::pair<mesh::CellType, int> XDMFFile::read_cell_type(std::string grid_name,
                                                        std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");
  pugi::xml_node grid_node
      = node.select_node(("Grid[@Name='" + grid_name + "']").c_str()).node();
  if (!grid_node)
    throw std::runtime_error("<Grid> with name '" + grid_name + "' not found.");

  // Get topology node
  pugi::xml_node topology_node = grid_node.child("Topology");
  assert(topology_node);

  // Get cell type
  const std::pair<std::string, int> cell_type_str
      = xdmf_utils::get_cell_type(topology_node);

  // Get topological dimensions
  mesh::CellType cell_type = mesh::to_type(cell_type_str.first);

  return {cell_type, cell_type_str.second};
}
//-----------------------------------------------------------------------------
void XDMFFile::write_information(std::string name, std::string value,
                                 std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");

  pugi::xml_node info_node = node.append_child("Information");
  assert(info_node);
  info_node.append_attribute("Name") = name.c_str();
  info_node.append_attribute("Value") = value.c_str();

  // Save XML file (on process 0 only)
  if (MPI::rank(_comm.comm()) == 0)
    _xml_doc->save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
std::string XDMFFile::read_information(std::string name, std::string xpath)
{
  pugi::xml_node node = _xml_doc->select_node(xpath.c_str()).node();
  if (!node)
    throw std::runtime_error("XML node '" + xpath + "' not found.");
  pugi::xml_node info_node
      = node.select_node(("Information[@Name='" + name + "']").c_str()).node();
  if (!info_node)
    throw std::runtime_error("<Information> with name '" + name
                             + "' not found.");

  // Read data and trim any leading/trailing whitespace
  std::string value_str = info_node.attribute("Value").as_string();
  return value_str;
}
//-----------------------------------------------------------------------------
MPI_Comm XDMFFile::comm() const { return _comm.comm(); }
//-----------------------------------------------------------------------------
