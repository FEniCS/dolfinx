// Copyright (C) 2015 Chris N. Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#ifdef HAS_HDF5

#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <dolfin/log/log.h>
#include "XDMFxml.h"

using namespace dolfin;

//----------------------------------------------------------------------------
XDMFxml::XDMFxml(std::string filename): _filename(filename)
{
  // Do nothing
}
//----------------------------------------------------------------------------
XDMFxml::~XDMFxml()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XDMFxml::header()
{
  xml_doc.append_child(pugi::node_doctype).set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");
  pugi::xml_node xdmf = xml_doc.append_child("Xdmf");
  xdmf.append_attribute("Version") = "2.0";
  xdmf.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
  xdmf.append_child("Domain");
}
//----------------------------------------------------------------------------
void XDMFxml::write() const
{
  // Write XML file
  xml_doc.save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void XDMFxml::read()
{
  pugi::xml_parse_result result = xml_doc.load_file(_filename.c_str());
  if (!result)
  {
    dolfin_error("XDMFxml.cpp",
                 "read mesh from XDMF/H5 files",
                 "XML parsing error.");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::string> XDMFxml::topology_name() const
{
  // Topology - check format and get dataset name
  pugi::xml_node xdmf_topology
    = xml_doc.child("Xdmf").child("Domain")
             .child("Grid").child("Topology");
  pugi::xml_node xdmf_topology_data = xdmf_topology.child("DataItem");

  if (!xdmf_topology or !xdmf_topology_data)
  {
    dolfin_error("XDMFxml.cpp",
                 "read mesh from XDMF/H5 files",
                 "XML parsing error. XDMF file should contain only one mesh/dataset");
  }

  const std::string
    topological_data_format(xdmf_topology_data.attribute("Format").value());
  if (topological_data_format != "HDF")
  {
    dolfin_error("XDMFxml.cpp",
                 "read mesh from XDMF/H5 files",
                 "XML parsing error. Wrong dataset format (not HDF5)");
  }

  // Usually, the DOLFIN CellType is just the lower case of the VTK name
  // FIXME: this will fail for 1D and quadratic topology
  std::string cell_type(xdmf_topology.attribute("TopologyType").value());
  boost::to_lower(cell_type);

  const std::string topo_ref(xdmf_topology_data.first_child().value());
  std::vector<std::string> topo_vec;
  boost::split(topo_vec, topo_ref, boost::is_any_of(":"));
  dolfin_assert(topo_vec.size() == 2);

  // Add cell type to topology data
  topo_vec.push_back(cell_type);

  return topo_vec;
}
//-----------------------------------------------------------------------------
std::vector<std::string> XDMFxml::geometry_name() const
{
  // Geometry - check format and get dataset name
  pugi::xml_node xdmf_geometry_data =
    xml_doc.child("Xdmf").child("Domain").child("Grid").child("Geometry").child("DataItem");
  dolfin_assert(xdmf_geometry_data);

  const std::string geom_fmt(xdmf_geometry_data.attribute("Format").value());
  if (geom_fmt != "HDF")
  {
    dolfin_error("XDMFxml.cpp",
                 "read mesh from XDMF/H5 files",
                 "XML parsing error. Wrong dataset format (not HDF5)");
  }

  const std::string geom_ref(xdmf_geometry_data.first_child().value());

  std::vector<std::string> geom_vec;
  boost::split(geom_vec, geom_ref, boost::is_any_of(":"));
  dolfin_assert(geom_vec.size() == 2);
  return geom_vec;
}
//-----------------------------------------------------------------------------
std::string XDMFxml::dataname() const
{
  // Values - check format and get dataset name
  pugi::xml_node xdmf_values = xml_doc.child("Xdmf").child("Domain").
    child("Grid").child("Grid").child("Attribute").child("DataItem");
  dolfin_assert(xdmf_values);

  const std::string value_fmt(xdmf_values.attribute("Format").value());
  dolfin_assert(value_fmt == "HDF");

  const std::string value_ref(xdmf_values.first_child().value());
  std::vector<std::string> value_bits;
  boost::split(value_bits, value_ref, boost::is_any_of(":/"));
  dolfin_assert(value_bits.size() == 5);
  dolfin_assert(value_bits[2] == "Mesh");
  dolfin_assert(value_bits[4] == "values");

  return value_bits[3];
}
//-----------------------------------------------------------------------------
void XDMFxml::data_attribute(std::string name,
                             std::size_t value_rank,
                             bool vertex_data,
                             std::size_t num_total_vertices,
                             std::size_t num_global_cells,
                             std::size_t padded_value_size,
                             std::string dataset_name)
{
  // Grid/Attribute (Function value data)
  pugi::xml_node xdmf_values = xdmf_grid.append_child("Attribute");
  xdmf_values.append_attribute("Name") = name.c_str();

  dolfin_assert(value_rank < 3);
  // 1D Vector should be treated as a Scalar
  std::size_t apparent_value_rank = (padded_value_size == 1) ? 0 : value_rank;
  static std::vector<std::string> attrib_type
    = {"Scalar", "Vector", "Tensor"};
  xdmf_values.append_attribute("AttributeType")
    = attrib_type[apparent_value_rank].c_str();

  xdmf_values.append_attribute("Center") = (vertex_data ? "Node" : "Cell");

  pugi::xml_node xdmf_data = xdmf_values.append_child("DataItem");
  xdmf_data.append_attribute("Format") = "HDF";

  const std::size_t num_total_entities
    = (vertex_data ? num_total_vertices : num_global_cells);
  const std::string s = std::to_string(num_total_entities) + " "
    + std::to_string(padded_value_size);
  xdmf_data.append_attribute("Dimensions") = s.c_str();

  xdmf_data.append_child(pugi::node_pcdata).set_value(dataset_name.c_str());
}
//-----------------------------------------------------------------------------
pugi::xml_node XDMFxml::init_mesh(std::string name)
{
  pugi::xml_node xdmf_domain = xml_doc.child("Xdmf").child("Domain");
  dolfin_assert(xdmf_domain);
  xdmf_grid = xdmf_domain.append_child("Grid");
  xdmf_grid.append_attribute("Name") = name.c_str();
  xdmf_grid.append_attribute("GridType") = "Uniform";
  xdmf_grid.append_child("Topology");
  xdmf_grid.append_child("Geometry");
  return xdmf_grid;
}
//-----------------------------------------------------------------------------
pugi::xml_node XDMFxml::init_timeseries(std::string name, double time_step,
                                        std::size_t counter)
 {
   if (counter == 0)
   {
     // First time step - create document template
     header();
   }
   else
   {
     // Subsequent timestep - read in existing XDMF file
     pugi::xml_parse_result result = xml_doc.load_file(_filename.c_str());
     if (!result)
     {
       dolfin_error("XDMFxml.cpp",
                    "write data to XDMF file",
                    "XML parsing error when reading from existing file");
     }
   }

   pugi::xml_node xdmf_domain = xml_doc.child("Xdmf").child("Domain");
   dolfin_assert(xdmf_domain);

   // Look for existing TimeSeries with same name
   pugi::xml_node xdmf_timegrid = xdmf_domain.first_child();

   // If not found, create a new TimeSeries
   pugi::xml_node xdmf_timedata;
   if (!xdmf_timegrid)
   {
     //  /Xdmf/Domain/Grid - actually a TimeSeries, not a spatial grid
     xdmf_timegrid = xdmf_domain.append_child("Grid");
     xdmf_timegrid.append_attribute("Name") = "TimeSeries";
     xdmf_timegrid.append_attribute("GridType") = "Collection";
     xdmf_timegrid.append_attribute("CollectionType") = "Temporal";

     //  /Xdmf/Domain/Grid/Time
     pugi::xml_node xdmf_time = xdmf_timegrid.append_child("Time");
     xdmf_time.append_attribute("TimeType") = "List";
     xdmf_timedata = xdmf_time.append_child("DataItem");
     xdmf_timedata.append_attribute("Format") = "XML";
     xdmf_timedata.append_attribute("Dimensions") = "0";
     xdmf_timedata.append_child(pugi::node_pcdata);
   }

   dolfin_assert(xdmf_timegrid);

   // Get time series node
   xdmf_timedata = xdmf_timegrid.child("Time").child("DataItem");
   dolfin_assert(xdmf_timedata);

   unsigned int last_count
     = std::stoul(xdmf_timedata.attribute("Dimensions").value());

   std::string times_str(xdmf_timedata.first_child().value());
   const std::string timestep_str
     = boost::str((boost::format("%g") % time_step));
   if (last_count != 0)
   {
     // Find last space character and last time stamp
     const std::size_t p = times_str.rfind(" ");
     dolfin_assert(p != std::string::npos);
     const std::string last_stamp(times_str.begin() + p + 1, times_str.end());

     if (timestep_str == last_stamp)
     {
       // Retrieve last "grid"
       xdmf_grid = xdmf_timegrid.last_child();
       return xdmf_grid;
     }
   }

   times_str += " " + timestep_str;
   ++last_count;

   xdmf_timedata.attribute("Dimensions").set_value(last_count);
   xdmf_timedata.first_child().set_value(times_str.c_str());

   //   /Xdmf/Domain/Grid/Grid - the actual data for this timestep
   xdmf_grid = xdmf_timegrid.append_child("Grid");
   std::string s = "grid_" + std::to_string(last_count);
   xdmf_grid.append_attribute("Name") = s.c_str();
   xdmf_grid.append_attribute("GridType") = "Uniform";

   // Grid/Topology
   pugi::xml_node topology = xdmf_grid.child("Topology");
   if (!topology)
     xdmf_grid.append_child("Topology");

   // Grid/Geometry
   pugi::xml_node geometry = xdmf_grid.child("Geometry");
   if (!geometry)
     xdmf_grid.append_child("Geometry");

   return xdmf_grid;
 }
//-----------------------------------------------------------------------------
void XDMFxml::mesh_topology(const CellType::Type cell_type,
                            const std::size_t cell_order,
                            const std::size_t num_global_cells,
                            const std::string reference)
{
  pugi::xml_node xdmf_topology = xdmf_grid.child("Topology");
  pugi::xml_node xdmf_topology_data = xdmf_topology.child("DataItem");

  // Check if already has topology data, in which case ignore
  if (xdmf_topology_data)
    return;

  xdmf_topology.append_attribute("NumberOfElements")
    = (unsigned int) num_global_cells;

  std::unique_ptr<CellType> celltype(CellType::create(cell_type));
  std::size_t nodes_per_element = celltype->num_entities(0);

  // Cell type
  if (cell_type == CellType::Type::point)
  {
    xdmf_topology.append_attribute("TopologyType") = "PolyVertex";
    xdmf_topology.append_attribute("NodesPerElement") = "1";
  }
  else if (cell_type == CellType::Type::interval and cell_order == 1)
  {
    xdmf_topology.append_attribute("TopologyType") = "PolyLine";
    xdmf_topology.append_attribute("NodesPerElement") = "2";
  }
  else if (cell_type == CellType::Type::interval and cell_order == 2)
  {
    xdmf_topology.append_attribute("TopologyType") = "Edge_3";
    nodes_per_element = 3;
  }
  else if (cell_type == CellType::Type::triangle and cell_order == 1)
    xdmf_topology.append_attribute("TopologyType") = "Triangle";
  else if (cell_type == CellType::Type::triangle and cell_order == 2)
  {
    xdmf_topology.append_attribute("TopologyType") = "Tri_6";
    nodes_per_element = 6;
  }
  else if (cell_type == CellType::Type::quadrilateral and cell_order == 1)
    xdmf_topology.append_attribute("TopologyType") = "Quadrilateral";
  else if (cell_type == CellType::Type::tetrahedron and cell_order == 1)
    xdmf_topology.append_attribute("TopologyType") = "Tetrahedron";
  else if (cell_type == CellType::Type::tetrahedron and cell_order == 2)
  {
    xdmf_topology.append_attribute("TopologyType") = "Tet_10";
    nodes_per_element = 10;
  }
  else if (cell_type == CellType::Type::hexahedron and cell_order == 1)
    xdmf_topology.append_attribute("TopologyType") = "Hexahedron";
  else
  {
    dolfin_error("XDMFFile.cpp",
                 "output mesh topology",
                 "Invalid combination of cell type and order");
  }

  if (reference.size() > 0)
  {
    // Refer to all cells and dimensions
    xdmf_topology_data = xdmf_topology.append_child("DataItem");
    xdmf_topology_data.append_attribute("Format") = "HDF";
    const std::string cell_dims = std::to_string(num_global_cells)
      + " " + std::to_string(nodes_per_element);
    xdmf_topology_data.append_attribute("Dimensions") = cell_dims.c_str();

    const std::string topology_reference = reference + "/topology";
    xdmf_topology_data.append_child(pugi::node_pcdata)
      .set_value(topology_reference.c_str());
  }
}
//----------------------------------------------------------------------------
void XDMFxml::mesh_geometry(const std::size_t num_total_vertices,
                            const std::size_t gdim,
                            const std::string reference)
{
  pugi::xml_node xdmf_geometry = xdmf_grid.child("Geometry");
  pugi::xml_node xdmf_geom_data = xdmf_geometry.child("DataItem");

  // Check if already has topology data, in which case ignore
  if (xdmf_geom_data)
    return;

  dolfin_assert(gdim > 0 && gdim <= 3);
  std::string geometry_type;
  if (gdim == 1)
  {
    // geometry "X" is not supported in XDMF
    geometry_type = "X_Y_Z";
  }
  else if (gdim == 2)
    geometry_type = "XY";
  else if (gdim == 3)
    geometry_type = "XYZ";

  xdmf_geometry.append_attribute("GeometryType") = geometry_type.c_str();
  xdmf_geom_data = xdmf_geometry.append_child("DataItem");

  xdmf_geom_data.append_attribute("Format") = "HDF";
  std::string geom_dim = std::to_string(num_total_vertices) + " "
    + std::to_string(gdim);
  xdmf_geom_data.append_attribute("Dimensions") = geom_dim.c_str();

  if (gdim == 1)
  {
    // FIXME: improve this workaround

    // When gdim==1, XDMF does not support a 1D geometry "X", so need
    // to provide some dummy Y and Z values.  Using the "X_Y_Z"
    // geometry the Y and Z values can be supplied as separate
    // datasets, here in plain text (though it could be done in HDF5
    // too).

    // Cannot write HDF5 here, as we are only running on rank 0, and
    // will deadlock.

    std::string dummy_zeros;
    dummy_zeros.reserve(2*num_total_vertices);
    for (std::size_t i = 0; i < num_total_vertices; ++i)
      dummy_zeros += "0 ";

    pugi::xml_node xdmf_geom_1 = xdmf_geometry.append_child("DataItem");
    xdmf_geom_1.append_attribute("Format") = "XML";
    geom_dim = std::to_string(num_total_vertices) + " 1" ;
    xdmf_geom_1.append_attribute("Dimensions") = geom_dim.c_str();
    xdmf_geom_1.append_child(pugi::node_pcdata).set_value(dummy_zeros.c_str());

    pugi::xml_node xdmf_geom_2 = xdmf_geometry.append_child("DataItem");
    xdmf_geom_2.append_attribute("Format") = "XML";
    geom_dim = std::to_string(num_total_vertices) + " 1" ;
    xdmf_geom_2.append_attribute("Dimensions") = geom_dim.c_str();
    xdmf_geom_2.append_child(pugi::node_pcdata).set_value(dummy_zeros.c_str());
  }

  const std::string geometry_reference = reference + "/coordinates";
  xdmf_geom_data.append_child(pugi::node_pcdata).set_value(geometry_reference.c_str());
}
//-----------------------------------------------------------------------------
#endif
