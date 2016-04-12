// Copyright (C) 2016 Quang T. Ha, Chris Richardson and Garth N. Wells
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

#include <string>
#include <sstream>
#include <utility>
#include "pugixml.hpp"

#include <dolfin/common/MPI.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Vertex.h>

#include "X3DOM.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
X3DOMParameters::X3DOMParameters()
  : _representation(Representation::surface_with_edges),
    _size({{500.0, 400.0}}),
    _show_viewpoints(true),
    _diffuse_color({{1.0, 1.0, 1.0}}),
    _emissive_color({{0.0, 0.0, 0.0}}),
    _specular_color({{0.0, 0.0, 0.0}}),
    _background_color({{0.95, 0.95, 0.95}}),
    _ambient_intensity(0.0),
    _shininess(0.5),
    _transparency(0.0),
    _show_x3d_stats(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_representation(Representation representation)
{
  _representation = representation;
}
//-----------------------------------------------------------------------------
X3DOMParameters::Representation X3DOMParameters::get_representation() const
{
  return _representation;
}
//-----------------------------------------------------------------------------
std::array<double, 2> X3DOMParameters::get_viewport_size() const
{
  return _size;
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_diffuse_color(std::array<double, 3> rgb)
{
  check_rgb(rgb);
  _diffuse_color = rgb;
}
//-----------------------------------------------------------------------------
std::array<double, 3> X3DOMParameters::get_diffuse_color() const
{
  return _diffuse_color;
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_emissive_color(std::array<double, 3> rgb)
{
  check_rgb(rgb);
  _emissive_color = rgb;
}
//-----------------------------------------------------------------------------
std::array<double, 3> X3DOMParameters::get_emissive_color() const
{
  return _emissive_color;
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_specular_color(std::array<double, 3> rgb)
{
  check_rgb(rgb);
  _specular_color = rgb;
}
//-----------------------------------------------------------------------------
std::array<double, 3> X3DOMParameters::get_specular_color() const
{
  return _specular_color;
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_background_color(std::array<double, 3> rgb)
{
  check_rgb(rgb);
  _background_color = rgb;
}
//-----------------------------------------------------------------------------
std::array<double, 3> X3DOMParameters::get_background_color() const
{
  return _background_color;
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_ambient_intensity(double intensity)
{
  check_value_range(intensity, 0.0, 1.0);
  _ambient_intensity = intensity;
}
//-----------------------------------------------------------------------------
double X3DOMParameters::get_ambient_intensity() const
{
  return _ambient_intensity;
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_shininess(double shininess)
{
  check_value_range(shininess, 0.0, 1.0);
  _shininess = shininess;
}
//-----------------------------------------------------------------------------
double X3DOMParameters::get_shininess() const
{
  return _shininess;
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_transparency(double transparency)
{
  check_value_range(transparency, 0.0, 1.0);
  _transparency = transparency;
}
//-----------------------------------------------------------------------------
double X3DOMParameters::get_transparency() const
{
  return _transparency;
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_viewpoint_buttons(bool show)
{
  _show_viewpoints = show;
}
//-----------------------------------------------------------------------------
bool X3DOMParameters::get_viewpoint_buttons() const
{
  return _show_viewpoints;
}
//-----------------------------------------------------------------------------
void X3DOMParameters::set_x3d_stats(bool show_stats)
{
  _show_x3d_stats = show_stats;
}
//-----------------------------------------------------------------------------
bool X3DOMParameters::get_x3d_stats() const
{
  return _show_x3d_stats;
  }
//-----------------------------------------------------------------------------
void X3DOMParameters::check_rgb(std::array<double, 3>& rgb)
{
  for (auto c : rgb)
  {
    if (c < 0.0 or c > 1.0)
    {
      dolfin_error("X3DOM.cpp",
                   "check validity of RGB (color) values",
                   "RGB components must be between 0.0 and 1.0");
    }
  }
}
//-----------------------------------------------------------------------------
void X3DOMParameters::check_value_range(double value, double lower,
                                        double upper)
{
  dolfin_assert(lower < upper);
  if (value < lower or value > upper)
  {
    dolfin_error("X3DOM.cpp",
                 "check validity of X3D properties",
                 "Parameter outside of allowable range of (%f, %f)", lower,
                 upper);
  }
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
std::string X3DOM::str(const Mesh& mesh, X3DOMParameters parameters)
{
  // Build XML doc
  pugi::xml_document xml_doc;
  x3dom(xml_doc, mesh, {}, {}, parameters);

  // Return as string
  return to_string(xml_doc);
}
//-----------------------------------------------------------------------------
std::string X3DOM::xhtml(const Mesh& mesh, X3DOMParameters parameters)
{
  // Build XML doc
  pugi::xml_document xml_doc;
  xhtml(xml_doc, mesh, {}, {}, parameters);

  // Return as string
  return to_string(xml_doc);
}
//-----------------------------------------------------------------------------
void X3DOM::get_function_values(const Function& u,
                                std::vector<double>& vertex_values,
                                std::vector<double>& facet_values)
{
  // Get dofmap and mesh
  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  const std::size_t value_rank = u.value_rank();
  // Print warning for vector-valued functions and error for
  // higher tensors
  if (value_rank == 1)
    warning("X3DOM outputs scalar magnitude of vector field");
  else if (value_rank > 1)
  {
    dolfin_error("X3DOM.cpp",
                 "write X3D",
                 "Can only handle scalar and vector Functions");
  }

  // Test for cell-centred data
  const std::size_t tdim = mesh.topology().dim();
  std::size_t cell_based_dim = 1;
  for (std::size_t i = 0; i < value_rank; i++)
    cell_based_dim *= tdim;
  const bool vertex_data = !(dofmap.max_element_dofs() == cell_based_dim);

  if (vertex_data)
  {
    // Compute vertex data values
    u.compute_vertex_values(vertex_values, mesh);

    // Compute l2 norm for vector-valued problems
    if (value_rank == 1)
    {
      const std::size_t num_vertices = mesh.num_vertices();
      std::vector<double> magnitude(num_vertices);
      for (std::size_t i = 0; i < num_vertices ; ++i)
      {
        double val = 0.0;
        for (std::size_t j = 0; j < u.value_size() ; j++)
          val += vertex_values[i + j*num_vertices]*vertex_values[i + j*num_vertices];
        magnitude[i] = std::sqrt(val);
      }

      // Swap data_values and magnitude
      std::swap(vertex_values, magnitude);
    }
  }
  else
  {
    if (value_rank != 0)
    {
      dolfin_error("X3DOM.cpp",
                   "create X3DOM",
                   "Can only handle scalar cell-centered Function at present");
    }

    if (MPI::size(mesh.mpi_comm()) != 1)
    {
      dolfin_error("X3DOM.cpp",
                   "create X3DOM",
                   "Cell-centered data not supported in parallel");
    }

    // Get dofs for cell centered data
    std::vector<dolfin::la_index> dofs(mesh.num_cells());
    for (std::size_t i = 0; i != mesh.num_cells(); ++i)
    {
      // Get dof index of cell data
      dolfin_assert(dofmap.num_element_dofs(i) == 1);
      dofs[i] = dofmap.cell_dofs(i)[0];
    }

    // Get  values from vector
    std::vector<double> cell_values(dofs.size());
    dolfin_assert(u.vector());
    u.vector()->get_local(cell_values.data(), dofs.size(), dofs.data());

    // FIXME: this is inefficient and a bit random for interior facets
    // (which we don't need) - so needs a redesign.
    facet_values.resize(mesh.num_facets());
    for (FaceIterator f(mesh); !f.end(); ++f)
      facet_values[f->index()] = cell_values[f->entities(tdim)[0]];
  }
}
//-----------------------------------------------------------------------------
std::string X3DOM::str(const Function& u, X3DOMParameters parameters)
{
  // Get values on vertices or facets
  std::vector<double> vertex_values;
  std::vector<double> facet_values;
  get_function_values(u, vertex_values, facet_values);

  // Get mesh
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  // Build XML doc
  pugi::xml_document xml_doc;
  x3dom(xml_doc, mesh, vertex_values, facet_values, parameters);

  // Return as string
  return to_string(xml_doc);
}
//-----------------------------------------------------------------------------
std::string X3DOM::xhtml(const Function& u, X3DOMParameters parameters)
{
  // Get values on vertices or facets
  std::vector<double> vertex_values;
  std::vector<double> facet_values;
  get_function_values(u, vertex_values, facet_values);

  // Get mesh
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  // Build XML doc
  pugi::xml_document xml_doc;
  xhtml(xml_doc, mesh, vertex_values, facet_values, parameters);

  // Return as string
  return to_string(xml_doc);
}
//-----------------------------------------------------------------------------
void X3DOM::x3dom(pugi::xml_document& xml_doc, const Mesh& mesh,
                  const std::vector<double>& vertex_values,
                  const std::vector<double>& facet_values,
                  const X3DOMParameters& parameters)
{
  // Build X3D XML and add to XML doc
  add_x3dom_data(xml_doc, mesh, vertex_values, facet_values, parameters);
}
//-----------------------------------------------------------------------------
void X3DOM::xhtml(pugi::xml_document& xml_doc, const Mesh& mesh,
                  const std::vector<double>& vertex_values,
                  const std::vector<double>& facet_values,
                  const X3DOMParameters& parameters)
{
  // Add DOCTYPE
  add_html_doctype(xml_doc);

  // Create 'html' node and add HTML preamble
  pugi::xml_node html_node = add_html_preamble(xml_doc);

  // Add body node
  pugi::xml_node body_node = html_node.append_child("body");

  // Add X3D XML data to 'body' node
  add_x3dom_data(body_node, mesh, vertex_values, facet_values, parameters);

  // Add text mesh info to body node
  pugi::xml_node mesh_info = body_node.append_child("div");
  dolfin_assert(mesh_info);
  mesh_info.append_attribute("style") = "position: absolute; margin: 1%; text-align: left; font-size: 12px; color: black;";
  std::string data = "Number of vertices: "
    + std::to_string(mesh.num_vertices())
    + ", number of cells: " + std::to_string(mesh.num_cells());
  mesh_info.append_child(pugi::node_pcdata).set_value(data.c_str());

  // FIXME: adding viewpoint buttons here is fragile since to may
  // change in the X3 node. Need to bring the two closer in the code
  // to keep consistency

  //Append viewpoint buttons to 'body' (HTML) node
  if (parameters.get_viewpoint_buttons())
  {
    // Add viewpoint control node to HTML
    pugi::xml_node viewpoint_control_node = body_node.append_child("div");

    // Add attributes to viewpoint node
    viewpoint_control_node.append_attribute("id") = "camera_buttons";
    viewpoint_control_node.append_attribute("style") = "display: block; position: relative; left: 1%";

    // Add viewpoints
    std::vector<std::string> viewpoints = {"front", "back", "left",
                                           "right", "top", "bottom"};
    for (auto viewpoint : viewpoints)
      add_viewpoint_control_option(viewpoint_control_node, viewpoint);
  }
}
//-----------------------------------------------------------------------------
pugi::xml_node X3DOM::add_html_preamble(pugi::xml_node& xml_node)
{
  // Add html node (required for XHTML)
  pugi::xml_node html_node = xml_node.append_child("html");
  html_node.append_attribute("xmlns") = "http://www.w3.org/1999/xhtml";
  html_node.append_attribute("lang") = "en";

  // Add head node (required for XHTML)
  pugi::xml_node head_node = html_node.append_child("head");

  // Add meta node(s)
  pugi::xml_node meta_node0 = head_node.append_child("meta");
  dolfin_assert(meta_node0);
  meta_node0.append_attribute("http-equiv") = "content-type";
  meta_node0.append_attribute("content") = "text/html;charset=UTF-8";

  pugi::xml_node meta_node1 = head_node.append_child("meta");
  dolfin_assert(meta_node1);
  meta_node1.append_attribute("name") = "generator";
  meta_node1.append_attribute("content")
    = "FEniCS/DOLFIN (http://fenicsproject.org)";


  // Add title node
  pugi::xml_node title_node = head_node.append_child("title");
  title_node.append_child(pugi::node_pcdata).set_value("FEniCS/DOLFIN X3DOM plot");

  // Add script node
  pugi::xml_node script_node = head_node.append_child("script");

  // Set attributes for script node
  script_node.append_attribute("type") = "text/javascript";
  script_node.append_attribute("src") = "http://www.x3dom.org/download/x3dom.js";
  script_node.append_child(pugi::node_pcdata);

  // Add link node
  pugi::xml_node link_node = head_node.append_child("link");

  // Set attributes for link node
  link_node.append_attribute("rel") = "stylesheet";
  link_node.append_attribute("type") = "text/css";
  link_node.append_attribute("href") = "http://www.x3dom.org/download/x3dom.css";

  return html_node;
}
//-----------------------------------------------------------------------------
void X3DOM::add_x3dom_doctype(pugi::xml_node& xml_node)
{
  dolfin_assert(xml_node);
  pugi::xml_node doc_type = xml_node.prepend_child(pugi::node_doctype);
  doc_type.set_value("X3D PUBLIC \"ISO//Web3D//DTD X3D 3.2//EN\" \"http://www.web3d.org/specifications/x3d-3.2.dtd\"");
}
//-----------------------------------------------------------------------------
void X3DOM::add_html_doctype(pugi::xml_node& xml_node)
{
  dolfin_assert(xml_node);
  pugi::xml_node doc_type = xml_node.prepend_child(pugi::node_doctype);
  doc_type.set_value("html PUBLIC \"-//W3C//DTD XHTML 1.0 Strict//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd\"");
}
//-----------------------------------------------------------------------------
pugi::xml_node X3DOM::add_x3d_node(pugi::xml_node& xml_node,
                                   std::array<double, 2> size, bool show_stats)
{
  pugi::xml_node x3d = xml_node.append_child("X3D");
  dolfin_assert(x3d);

  // Add on option to show rendering
  x3d.append_attribute("showStat") = show_stats;

  //x3d.append_attribute("profile") = "Interchange";
  //x3d.append_attribute("version") = "3.3";
  x3d.append_attribute("xmlns")
    = "http://www.web3d.org/specifications/x3d-namespace";
  //x3d.append_attribute("xsd:noNamespaceSchemaLocation")
  //  = "http://www.web3d.org/specifications/x3d-3.2.xsd";

  std::string width = std::to_string(size[0]) + "px";
  std::string height = std::to_string(size[1]) + "px";
  x3d.append_attribute("width") = width.c_str();
  x3d.append_attribute("height") = height.c_str();

  return x3d;
}
//-----------------------------------------------------------------------------
void X3DOM::add_x3dom_data(pugi::xml_node& xml_node, const Mesh& mesh,
                           const std::vector<double>& vertex_data,
                           const std::vector<double>& facet_data,
                           const X3DOMParameters& parameters)
{
  // Check that mesh is embedded in 2D or 3D
  const std::size_t gdim = mesh.geometry().dim();
  if (gdim != 2 and gdim != 3)
  {
    dolfin_error("X3DOM.cpp",
                 "get X3DOM string representation of a mesh",
                 "X3D works only for 2D and 3D meshes");
  }

  // X3D doctype
  add_x3dom_doctype(xml_node);

  // Add X3D node
  pugi::xml_node x3d_node = add_x3d_node(xml_node, parameters.get_viewport_size(),
                                         parameters.get_x3d_stats());
  dolfin_assert(x3d_node);

  // Add head node
  pugi::xml_node head_node = x3d_node.append_child("head");
  dolfin_assert(head_node);

  // Add meta node to head node, and attributes
  pugi::xml_node meta_node = head_node.append_child("meta");
  dolfin_assert(meta_node);
  meta_node.append_attribute("name") = "generator";
  meta_node.append_attribute("content")
    = "FEniCS/DOLFIN (http://fenicsproject.org)";

  // Add scene node
  pugi::xml_node scene = x3d_node.append_child("Scene");
  dolfin_assert(scene);

  // Add mesh to 'scene' XML node
  auto representation = parameters.get_representation();
  if (representation == X3DOMParameters::Representation::surface_with_edges)
  {
    // Add surface and then wireframe
    add_mesh_data(scene, mesh, vertex_data, facet_data, parameters, true);
    add_mesh_data(scene, mesh, {}, {}, parameters, false);
  }
  else if (representation == X3DOMParameters::Representation::surface)
    add_mesh_data(scene, mesh, vertex_data, facet_data, parameters, true);
  else
    add_mesh_data(scene, mesh, {}, {}, parameters, false);

  // Add viewpoint(s)
  const std::pair<Point, double> position = mesh_min_max(mesh);
  add_viewpoint_nodes(scene, position.first, position.second,
                     parameters.get_viewpoint_buttons());

  // Add background color
  pugi::xml_node background = scene.append_child("Background");
  background.append_attribute("skyColor")
    = array_to_string3(parameters.get_background_color()).c_str();

  // Add ambient light
  pugi::xml_node ambient_light_node = scene.append_child("DirectionalLight");
  ambient_light_node.append_attribute("ambientIntensity")
    = parameters.get_ambient_intensity();
  ambient_light_node.append_attribute("intensity") = 1.0;
}
//-----------------------------------------------------------------------------
void X3DOM::add_mesh_data(pugi::xml_node& xml_node, const Mesh& mesh,
                          const std::vector<double>& vertex_values,
                          const std::vector<double>& facet_values,
                          const X3DOMParameters& parameters,
                          bool surface)
{
  // X3DOM string for surface/wireframe
  const std::string x3d_type = surface ? "IndexedFaceSet" : "IndexedLineSet";

  // Get mesh topology and geometry data, and vertex/facet values
  std::vector<int> topology_data;
  std::vector<double> geometry_data;
  std::vector<double> values_data;
  build_mesh_data(topology_data, geometry_data, values_data,
                  mesh, vertex_values, facet_values, surface);

  // Add data to XML tree (on root process only)
  if (dolfin::MPI::rank(mesh.mpi_comm()) == 0)
  {
    // Create shape node and append ID attribute
    pugi::xml_node shape_node = xml_node.append_child("Shape");

    // Create appearance node
    pugi::xml_node appearance_node = shape_node.append_child("Appearance");

    // Append color attributes
    pugi::xml_node material_node = appearance_node.append_child("Material");
    if (surface)
    {
      material_node.append_attribute("diffuseColor")
        = array_to_string3(parameters.get_diffuse_color()).c_str();
    }

    material_node.append_attribute("emissiveColor")
      = array_to_string3(parameters.get_emissive_color()).c_str();
    material_node.append_attribute("specularColor")
      = array_to_string3(parameters.get_specular_color()).c_str();

    // Append ??? attributes
    //
    material_node.append_attribute("ambientIntensity") = parameters.get_ambient_intensity();
    material_node.append_attribute("shininess") = parameters.get_shininess();
    material_node.append_attribute("transparency") = parameters.get_transparency();

    // Add edges node
    pugi::xml_node indexed_set = shape_node.append_child(x3d_type.c_str());
    indexed_set.append_attribute("solid") = "false";

    // Add color per vertex attribute
    const bool color_per_vertex = !vertex_values.empty();
    indexed_set.append_attribute("colorPerVertex") = color_per_vertex;

    // Add topology data to edges node
    std::stringstream topology_str;
    for (auto c : topology_data)
      topology_str << c << " ";
    indexed_set.append_attribute("coordIndex") = topology_str.str().c_str();

    // Add Coordinate node
    pugi::xml_node coordinate_node = indexed_set.append_child("Coordinate");
    coordinate_node.append_attribute("DEF") = "dolfin";

    // Add geometry data to coordinate node
    std::stringstream geometry_str;
    for (auto x : geometry_data)
      geometry_str << x << " ";
    coordinate_node.append_attribute("point") = geometry_str.str().c_str();

    if (!values_data.empty())
    {
      if (color_per_vertex)
        dolfin_assert(3*values_data.size() == geometry_data.size());
      else
        dolfin_assert(4*values_data.size() == topology_data.size());

      // Get min/max values
      const double value_min = *std::min_element(values_data.begin(),
                                                 values_data.end());
      const double value_max = *std::max_element(values_data.begin(),
                                                 values_data.end());

      const double scale = (value_max == value_min) ? 1.0 : 255.0/(value_max - value_min);

      // Get colour map (256 RGB values)
      boost::multi_array<float, 2> cmap = color_map();

      // Add vertex colors
      std::stringstream color_values;
      for (auto x : values_data)
      {
        const int cindex = scale*std::abs(x - value_min);
        dolfin_assert(cindex < (int)cmap.shape()[0]);
        auto color = cmap[cindex];
        dolfin_assert(color.shape()[0] == 3);
        color_values << color[0] << " " << color[1] << " " << color[2] << " ";
      }

      pugi::xml_node color_node = indexed_set.append_child("Color");
      dolfin_assert(color_node);
      color_node.append_attribute("color") = color_values.str().c_str();
    }
  }

}
//-----------------------------------------------------------------------------
void X3DOM::add_viewpoint_control_option(pugi::xml_node& viewpoint_control,
                                         std::string viewpoint_label)
{
  std::string onclick_str
    = "document.getElementById('" + viewpoint_label + "').setAttribute('set_bind','true');";
  pugi::xml_node viewpoint_buttons = viewpoint_control.append_child("button");
  viewpoint_buttons.append_attribute("onclick") = onclick_str.c_str();
  viewpoint_buttons.append_attribute("style") = "display: block; margin: 2%";
  viewpoint_buttons.append_child(pugi::node_pcdata).set_value(viewpoint_label.c_str());
}
//-----------------------------------------------------------------------------
void X3DOM::add_viewpoint_nodes(pugi::xml_node& xml_scene,
                                const Point p, double d,
                                bool show_viewpoint_buttons)
{
  // Default viewpoint
  add_viewpoint_node(xml_scene, Viewpoint::default_view, p, d);

  // Add viewpoint nodes
  if (show_viewpoint_buttons)
  {
    add_viewpoint_node(xml_scene, Viewpoint::top, p, d);
    add_viewpoint_node(xml_scene, Viewpoint::bottom, p, d);
    add_viewpoint_node(xml_scene, Viewpoint::left, p, d);
    add_viewpoint_node(xml_scene, Viewpoint::right, p, d);
    add_viewpoint_node(xml_scene, Viewpoint::back, p, d);
    add_viewpoint_node(xml_scene, Viewpoint::front, p, d);
  }
}
//-----------------------------------------------------------------------------
void X3DOM::add_viewpoint_node(pugi::xml_node& xml_scene, Viewpoint viewpoint,
                               const Point p, const double d)
{
  std::string viewpoint_str;
  std::string orientation;
  std::string position;

  // Set viewpoint, orientation and position strings
  switch (viewpoint)
  {
  case Viewpoint::top:
    viewpoint_str = "top";
    orientation = "-1 0 0 1.5707963267948";
    position = array_to_string3({{p[0], p[1] + d - p[2], p[2]}});
    break;
  case Viewpoint::bottom:
    viewpoint_str = "bottom";
    orientation = "1 0 0 1.5707963267948";
    position = array_to_string3({{p[0], p[1] - d + p[2], p[2]}});
    break;
  case Viewpoint::left:
    viewpoint_str = "left";
    orientation = "0 1 0 1.5707963267948";
    position = array_to_string3({{p[0] + d - p[2], p[1], p[2]}});
    break;
  case Viewpoint::right:
    viewpoint_str = "right";
    orientation = "0 -1 0 1.5707963267948";
    position = array_to_string3({{p[0] - d + p[2], p[1], p[2]}});
    break;
  case Viewpoint::back:
    viewpoint_str = "back";
    orientation = "0 1 0 3.1415926535898";
    position = array_to_string3({{p[0], p[1], p[2] - d}});
    break;
  case Viewpoint::front:
    viewpoint_str = "front";
    orientation = "0 0 0 1";
    position = array_to_string3({{p[0], p[1], d}});
    break;
  case Viewpoint::default_view:
    viewpoint_str = "default";
    orientation = "-0.7071067812 0.7071067812 0 1";
    position = array_to_string3({{p[0] + 0.7071067812*(d - p[2]),
            p[1] + 0.7071067812*(d - p[2]),
            p[2] + 0.7071067812*(d - p[2])}});
    break;
  default:
    dolfin_error("X3DOM.cpp",
                 "add viewpoint node",
                 "Unknown Viewpoint enum");
    break;
  }

  // Append the node
  pugi::xml_node viewpoint_node = xml_scene.append_child("Viewpoint");

  // Add attributes to node
  viewpoint_node.append_attribute("id") = viewpoint_str.c_str();
  viewpoint_node.append_attribute("position") = position.c_str();

  viewpoint_node.append_attribute("orientation") = orientation.c_str();
  viewpoint_node.append_attribute("fieldOfView") = "0.785398";
  viewpoint_node.append_attribute("centerOfRotation")
    = array_to_string3({{p[0], p[1], p[2]}}).c_str();

  viewpoint_node.append_attribute("zNear") = "-1";
  viewpoint_node.append_attribute("zFar") = "-1";
}
//-----------------------------------------------------------------------------
boost::multi_array<float, 2> X3DOM::color_map()
{
  boost::multi_array<float, 2> rgb_map(boost::extents[256][3]);

  // Create RGB palette of 256 colors
  for (int i = 0; i < 256; ++i)
  {
    const double x = (double)i/255.0;
    const double y = 1.0 - x;
    rgb_map[i][0] = 4*std::pow(x, 3) - 3*std::pow(x, 4);
    rgb_map[i][1] = 4*std::pow(x, 2)*(1.0 - std::pow(x, 2));
    rgb_map[i][2] = 4*std::pow(y, 3) - 3*std::pow(y, 4);
  }

  /*
  for (int i = 0; i < 256; ++i)
  {
    const double lm = 425.0 + 250.0*(double)i/255.0;
    const double b
      = 1.8*exp(-std::pow((lm - 450.0)/((lm > 450.0) ? 40.0 : 20.0), 2.0));
    const double g
      = 0.9*exp(-std::pow((lm - 550.0)/((lm > 550.0) ? 60 : 40.0), 2.0));
    double r = 1.0*exp(-std::pow((lm - 600.0)/((lm > 600.0) ? 40.0 : 50.0), 2.0));
    r += 0.3*exp(-std::pow((lm - 450.0)/((lm > 450.0) ? 20.0 : 30.0), 2.0));
    const double total = r + g + b;

    rgb_map[i][0] = r/total;
    rgb_map[i][1] = g/total;
    rgb_map[i][2] = b/total;
  }
  */

  return rgb_map;
}
//-----------------------------------------------------------------------------
std::pair<Point, double> X3DOM::mesh_min_max(const Mesh& mesh)
{
  // Get dimensions
  double xmin = std::numeric_limits<double>::max();
  double xmax = std::numeric_limits<double>::min();
  double ymin = std::numeric_limits<double>::max();
  double ymax = std::numeric_limits<double>::min();
  double zmin = std::numeric_limits<double>::max();
  double zmax = std::numeric_limits<double>::min();

  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    const Point x = v->point();
    xmin = std::min(xmin, x[0]);
    xmax = std::max(xmax, x[0]);
    ymin = std::min(ymin, x[1]);
    ymax = std::max(ymax, x[1]);
    zmin = std::min(zmin, x[2]);
    zmax = std::max(zmax, x[2]);
  }

  // Get MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  xmin = dolfin::MPI::min(mpi_comm, xmin);
  ymin = dolfin::MPI::min(mpi_comm, ymin);
  zmin = dolfin::MPI::min(mpi_comm, zmin);
  xmax = dolfin::MPI::max(mpi_comm, xmax);
  ymax = dolfin::MPI::max(mpi_comm, ymax);
  zmax = dolfin::MPI::max(mpi_comm, zmax);

  // Compute midpoint of mesh
  Point midpoint((xmax + xmin)/2.0, (ymax + ymin)/2.0, (zmax + zmin)/2.0);

  // FIXME: explain this
  double d = std::max(xmax - xmin, ymax - ymin);
  d = 2.0*std::max(d, zmax - zmin) + zmax ;

  return {midpoint, d};
}
//-----------------------------------------------------------------------------
std::set<int> X3DOM::surface_vertex_indices(const Mesh& mesh)
{
  // Get mesh toplogical dimension
  const std::size_t tdim = mesh.topology().dim();

  // Initialise connectivities
  mesh.init(tdim - 1);
  mesh.init(tdim - 1, 0);

  // Fill set of surface vertex indices
  std::set<int> vertex_set;
  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    // If in 3D, only output exterior faces
    // FIXME: num_global_entities not working in serial
    if (tdim == 2 or f->num_global_entities(tdim) == 1)
    {
      for (VertexIterator v(*f); !v.end(); ++v)
        vertex_set.insert(v->index());
    }
  }

  return vertex_set;
}
//-----------------------------------------------------------------------------
void X3DOM::build_mesh_data(std::vector<int>& topology,
                            std::vector<double>& geometry,
                            std::vector<double>& value_data,
                            const Mesh& mesh,
                            const std::vector<double>& vertex_values,
                            const std::vector<double>& facet_values,
                            bool surface)
{
  // Cannot build data from facets and vertices at the same time
  dolfin_assert(vertex_values.empty() or facet_values.empty());
  // FIXME: also build value_data from facet_values

  // Get topological dimension
  const std::size_t tdim = mesh.topology().dim();

  // Intialise facet-to-cell connectivity
  mesh.init(tdim - 1 , tdim);

  // Get vertex indices
  const std::set<int> vertex_indices = surface_vertex_indices(mesh);

  std::size_t offset = dolfin::MPI::global_offset(mesh.mpi_comm(),
                                                  vertex_indices.size(), true);

  // Collect up topology of the local part of the mesh which should be
  // displayed
  std::vector<int> local_topology;
  std::vector<double> local_values;
  std::vector<int> local_facet_indices;
  if (surface)
  {
    // Build face data structure
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (tdim == 2 or f->num_global_entities(tdim) == 1)
      {
        local_facet_indices.push_back(f->index());

        for (VertexIterator v(*f); !v.end(); ++v)
        {
          // Find position of vertex in set
          std::size_t index = v->index();
          std::size_t pos = std::distance(vertex_indices.begin(),
                                          vertex_indices.find(index));
          local_topology.push_back(pos + offset);
        }

        // Add -1 to denote end of entity
        local_topology.push_back(-1);
      }
    }
  }
  else
  {
    // Build edge data structure
    for (EdgeIterator e(mesh); !e.end(); ++e)
    {
      // For 3D, check if one of the faces connected to this edge is
      // external, in which case we add the edge
      bool add_edge = true;
      if (tdim == 3)
      {
        add_edge = false;
        for (FaceIterator f(*e); !f.end(); ++f)
        {
          if (f->num_global_entities(tdim) == 1)
            add_edge = true;
        }
      }

      // Add edge to data structure, if required
      if (add_edge)
      {
        for (VertexIterator v(*e); !v.end(); ++v)
        {
          // Find position of vertex in set
          std::size_t index = v->index();
          std::size_t pos = std::distance(vertex_indices.begin(),
                                          vertex_indices.find(index));
          local_topology.push_back(pos + offset);
        }

        // Add -1 to denote end of entity
        local_topology.push_back(-1);
      }
    }
  }

  // Gather up all topology on process 0
  dolfin::MPI::gather(mesh.mpi_comm(), local_topology, topology);

  // Collect up geometry of all local points
  std::vector<double> local_geometry;
  for (auto index : vertex_indices)
  {
    Vertex v(mesh, index);
    const Point p = v.point();
    for (std::size_t i = 0; i < 3; ++i)
      local_geometry.push_back(p[i]);

    // Add vertex data
    if (!vertex_values.empty())
      local_values.push_back(vertex_values[index]);
  }

  if (!facet_values.empty())
  {
    for (auto index : local_facet_indices)
      local_values.push_back(facet_values[index]);
  }

  // Gather up all geometry on process 0 and append to xml
  dolfin::MPI::gather(mesh.mpi_comm(), local_geometry, geometry);

  value_data = local_values;
}
//-----------------------------------------------------------------------------
std::string X3DOM::array_to_string3(std::array<double, 3> x)
{
  std::string str = std::to_string(x[0]) + " " + std::to_string(x[1])
    + " " + std::to_string(x[2]);

  return str;
}
//-----------------------------------------------------------------------------
std::string X3DOM::to_string(pugi::xml_document& xml_doc)
{
  // Save XML doc to stringstream
  std::stringstream s;
  const std::string indent = "  ";
  xml_doc.save(s, indent.c_str());

  // Return string
  return s.str();
}
//-----------------------------------------------------------------------------
