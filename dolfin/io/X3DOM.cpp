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

#include <dolfin/common/MPI.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Vertex.h>

#include "X3DOM.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::string X3DOM::str(const Mesh& mesh, X3DParameters parameters)
{
  // Create empty pugi XML doc
  pugi::xml_document xml_doc;

  // Build X3D XML and add to XML doc
  add_x3dom_data(xml_doc, mesh, parameters);

  // Save XML doc to stringstream
  std::stringstream s;
  const std::string indent = "  ";
  xml_doc.save(s, indent.c_str());

  // Return string
  return s.str();
}
//-----------------------------------------------------------------------------
std::string X3DOM::html(const Mesh& mesh, X3DParameters parameters)
{
  // Create empty pugi XML doc
  pugi::xml_document xml_doc;

  // Add doc style to enforce xhtml
  xml_doc.append_child("!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\"");

  // Add html node
  pugi::xml_node node = xml_doc.append_child("html");
  node.append_attribute("xmlns") = "http://www.w3.org/1999/xhtml";
  node.append_attribute("lang") = "en";

  // Add head node
  pugi::xml_node head = node.append_child("head");

  // Add meta node
  pugi::xml_node meta = head.append_child("meta");
  meta.append_attribute("http-equiv") = "content-type";
  meta.append_attribute("content") = "text/xhtml; charset=UTF-8";

  // Add script node
  pugi::xml_node script = head.append_child("script");

  // Set attributes for script node
  script.append_attribute("type") = "text/javascript";
  script.append_attribute("src") = "http://www.x3dom.org/download/x3dom.js";
  script.append_child(pugi::node_pcdata);

  // Add link node
  pugi::xml_node link = head.append_child("link");

  // Set attributes for link node
  link.append_attribute("rel") = "stylesheet";
  link.append_attribute("type") = "text/css";
  link.append_attribute("href") = "http://www.x3dom.org/download/x3dom.css";

  // Add body node
  pugi::xml_node body_node = node.append_child("body");

  // Add X3D XML data to 'body' node
  add_x3dom_data(body_node, mesh, parameters);

  // Append viewpoint buttons to 'body' node
  if (parameters.get_viewpoint_buttons())
  {
    // Add viewpoint control node
    pugi::xml_node viewpoint_control_node = body_node.append_child("div");

    // Add attributes to viewpoint niode
    viewpoint_control_node.append_attribute("id") = "camera_buttons";
    viewpoint_control_node.append_attribute("style") = "display: block";

    // Add viewpoints
    std::vector<std::string> viewpoints = {"front", "back", "left",
                                           "right", "top", "bottom"};
    for (auto viewpoint : viewpoints)
      add_viewpoint_control_option(viewpoint_control_node, viewpoint);
  }

  // Save XML doc to stringstream, without default XML header
  std::stringstream s;
  const std::string indent = "  ";
  xml_doc.save(s, indent.c_str(),
               pugi::format_default | pugi::format_no_declaration);

  // Return string
  return s.str();
}
//-----------------------------------------------------------------------------
void X3DOM::add_doctype(pugi::xml_node& xml_node)
{
  dolfin_assert(xml_node);
  pugi::xml_node doc_type = xml_node.prepend_child(pugi::node_doctype);
  doc_type.set_value("X3D PUBLIC \"ISO//Web3D//DTD X3D 3.2//EN\" \"http://www.web3d.org/specifications/x3d-3.2.dtd\"");
}
//-----------------------------------------------------------------------------
pugi::xml_node X3DOM::add_x3d_node(pugi::xml_node& xml_node)
{
  pugi::xml_node x3d = xml_node.append_child("X3D");
  dolfin_assert(x3d);

  // Add on option to show rendering
  x3d.append_attribute("showStat") = "true";

  x3d.append_attribute("profile") = "Interchange";
  x3d.append_attribute("version") = "3.2";
  x3d.append_attribute("xmlns:xsd")
    = "http://www.w3.org/2001/XMLSchema-instance";
  x3d.append_attribute("xsd:noNamespaceSchemaLocation")
    = "http://www.web3d.org/specifications/x3d-3.2.xsd";
  x3d.append_attribute("width") = "500px";
  x3d.append_attribute("height") = "400px";

  return x3d;
}
//-----------------------------------------------------------------------------
void X3DOM::add_x3dom_data(pugi::xml_node& xml_node, const Mesh& mesh,
                           const X3DParameters& parameters)
{
  // Check that mesh is embedded in 2D or 3D
  const std::size_t gdim = mesh.geometry().dim();
  if (gdim !=2 and gdim !=3)
  {
    dolfin_error("X3DOM.cpp",
                 "get X3DOM string representation of a mesh",
                 "X3D works only for 2D and 3D meshes");
  }

  // X3D doctype
  add_doctype(xml_node);

  // Add X3D node
  pugi::xml_node x3d_node = add_x3d_node(xml_node);
  dolfin_assert(x3d_node);

  // Intialise facet-to-cell connectivity
  const std::size_t tdim = mesh.geometry().dim();
  mesh.init(tdim - 1 , tdim);

  // Add scene node
  pugi::xml_node scene = x3d_node.append_child("Scene");
  dolfin_assert(scene);

  // Add mesh to 'scene' XML node
  auto representation = parameters.get_representation();
  if (representation == X3DParameters::Representation::surface_with_edges)
  {
    add_mesh_data(scene, mesh, parameters, true);
    add_mesh_data(scene, mesh, parameters, false);
  }
  else if (representation == X3DParameters::Representation::surface)
    add_mesh_data(scene, mesh, parameters, true);
  else
    add_mesh_data(scene, mesh, parameters, false);

  // Add viewpoint(s)
  const std::vector<double> xpos = mesh_min_max(mesh);
  add_viewpoint_nodes(scene, xpos, parameters.get_viewpoint_buttons());

  // Add background colour
  pugi::xml_node background = scene.append_child("Background");
  background.append_attribute("skyColor")
    = array_to_string3(parameters.get_background_colour()).c_str();

  // Add ambient light
  pugi::xml_node ambient_light = scene.append_child("DirectionalLight");
  ambient_light.append_attribute("ambientIntensity") = parameters.get_ambient_intensity();
  ambient_light.append_attribute("intensity") = "0";

  // Add text mesh info to X3D node
  pugi::xml_node mesh_info = x3d_node.append_child("div");
  dolfin_assert(mesh_info);
  mesh_info.append_attribute("style") = "position: absolute; bottom: 2%; left: 2%; text-align: left; font-size: 12px; color: black;";
  std::string data = "Number of vertices: "
    + std::to_string(mesh.num_vertices())
    + ", number of cells: " + std::to_string(mesh.num_cells());
  mesh_info.append_child(pugi::node_pcdata).set_value(data.c_str());
}
//-----------------------------------------------------------------------------
void X3DOM::add_mesh_data(pugi::xml_node& xml_node, const Mesh& mesh,
                          const X3DParameters& parameters,
                          bool surface)
{
  // X3DOM string for surface/wireframe
  const std::string x3d_type = surface ? "IndexedFaceSet" : "IndexedLineSet";

  // Get mesh topology and geometry data
  std::vector<int> topology;
  std::vector<double> geometry;
  build_mesh_data(topology, geometry, mesh, surface);

  // Add data to XML tree (on root process only)
  if (dolfin::MPI::rank(mesh.mpi_comm()) == 0)
  {
    // Create shape node and append ID attribute
    pugi::xml_node shape_node = xml_node.append_child("Shape");

    // Create appearance node
    pugi::xml_node appearance_node = shape_node.append_child("Appearance");

    // Append colour attributes
    pugi::xml_node material_node = appearance_node.append_child("Material");
    material_node.append_attribute("diffuseColor")
      = array_to_string3(parameters.get_diffuse_colour()).c_str();
    material_node.append_attribute("emmisiveColor")
      = array_to_string3(parameters.get_emmisive_colour()).c_str();
    material_node.append_attribute("specularColor")
      = array_to_string3(parameters.get_specular_colour()).c_str();

    // Append ??? attributes
    material_node.append_attribute("ambientIntensity") = parameters.get_ambient_intensity();
    material_node.append_attribute("shininess") = parameters.get_shininess();
    material_node.append_attribute("transparency") = parameters.get_transparency();

    // Add edges node
    pugi::xml_node indexed_face_set = shape_node.append_child(x3d_type.c_str());
    indexed_face_set.append_attribute("solid") = "false";

    // Add data to edges node
    std::stringstream topology_str;
    for (auto c : topology)
      topology_str << c << " ";
    indexed_face_set.append_attribute("coordIndex") = topology_str.str().c_str();

    // Add coordinate node
    pugi::xml_node coordinate = indexed_face_set.append_child("Coordinate");

    // Add data to coordinate node
    std::stringstream geometry_str;
    for (auto x : geometry)
      geometry_str << x << " ";
    coordinate.append_attribute("point") = geometry_str.str().c_str();
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
  viewpoint_buttons.append_attribute("style") = "display: block";
  viewpoint_buttons.append_child(pugi::node_pcdata).set_value(viewpoint_label.c_str());
}
//-----------------------------------------------------------------------------
void X3DOM::add_viewpoint_nodes(pugi::xml_node& xml_scene,
                                const std::vector<double>& xpos,
                                bool show_viewpoint_buttons)
{
  // Create center of rotation string
  std::string center_of_rotation = array_to_string3({xpos[0], xpos[1], xpos[2]});

  // Add viewpoint nodes
  if (show_viewpoint_buttons)
  {
    add_viewpoint_node(xml_scene, Viewpoint::top, center_of_rotation, xpos);
    add_viewpoint_node(xml_scene, Viewpoint::bottom, center_of_rotation, xpos);
    add_viewpoint_node(xml_scene, Viewpoint::left, center_of_rotation, xpos);
    add_viewpoint_node(xml_scene, Viewpoint::right, center_of_rotation, xpos);
    add_viewpoint_node(xml_scene, Viewpoint::back, center_of_rotation, xpos);
    add_viewpoint_node(xml_scene, Viewpoint::front, center_of_rotation, xpos);
  }

  // Default viewpoint
  add_viewpoint_node(xml_scene, Viewpoint::default_view, center_of_rotation,
                     xpos);
}
//-----------------------------------------------------------------------------
void X3DOM::add_viewpoint_node(pugi::xml_node& xml_scene, Viewpoint viewpoint,
                               const std::string center_of_rotation,
                               const std::vector<double>& xpos)
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
    position = array_to_string3({xpos[0], xpos[1] + xpos[3] - xpos[2], xpos[2]});
    break;
  case Viewpoint::bottom:
    viewpoint_str = "bottom";
    orientation = "1 0 0 1.5707963267948";
    position = array_to_string3({xpos[0], xpos[1] - xpos[3] + xpos[2], xpos[2]});
    break;
  case Viewpoint::left:
    viewpoint_str = "left";
    orientation = "0 1 0 1.5707963267948";
    position = array_to_string3({xpos[0] + xpos[3] - xpos[2], xpos[1], xpos[2]});
    break;
  case Viewpoint::right:
    viewpoint_str = "right";
    orientation = "0 -1 0 1.5707963267948";
    position = array_to_string3({xpos[0] - xpos[3] + xpos[2], xpos[1], xpos[2]});
    break;
  case Viewpoint::back:
    viewpoint_str = "back";
    orientation = "0 1 0 3.1415926535898";
    position = array_to_string3({xpos[0], xpos[1], xpos[2] - xpos[3]});
    break;
  case Viewpoint::front:
    viewpoint_str = "front";
    orientation = "0 0 0 1";
    position = array_to_string3({xpos[0], xpos[1], xpos[3]});
    break;
  case Viewpoint::default_view:
    viewpoint_str = "default";
    orientation = "-0.7071067812 0.7071067812 0 1";
    position = array_to_string3({xpos[0] + 0.7071067812*(xpos[3] - xpos[2]),
          xpos[1] + 0.7071067812*(xpos[3] - xpos[2]),
          xpos[2] + 0.7071067812*(xpos[3] - xpos[2])});
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
    = center_of_rotation.c_str();

  viewpoint_node.append_attribute("zNear") = "-1";
  viewpoint_node.append_attribute("zFar") = "-1";
}
//-----------------------------------------------------------------------------
std::vector<double> X3DOM::mesh_min_max(const Mesh& mesh)
{
  // Get MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

  // Get dimensions
  double xmin = std::numeric_limits<double>::max();
  double xmax = std::numeric_limits<double>::min();
  double ymin = std::numeric_limits<double>::max();
  double ymax = std::numeric_limits<double>::min();
  double zmin = std::numeric_limits<double>::max();
  double zmax = std::numeric_limits<double>::min();

  const std::size_t gdim = mesh.geometry().dim();
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    xmin = std::min(xmin, v->x(0));
    xmax = std::max(xmax, v->x(0));
    ymin = std::min(ymin, v->x(1));
    ymax = std::max(ymax, v->x(1));
    if (gdim == 2)
    {
      zmin = 0.0;
      zmax = 0.0;
    }
    else
    {
      zmin = std::min(zmin, v->x(2));
      zmax = std::max(zmax, v->x(2));
    }
  }

  xmin = dolfin::MPI::min(mpi_comm, xmin);
  ymin = dolfin::MPI::min(mpi_comm, ymin);
  zmin = dolfin::MPI::min(mpi_comm, zmin);

  xmax = dolfin::MPI::max(mpi_comm, xmax);
  ymax = dolfin::MPI::max(mpi_comm, ymax);
  zmax = dolfin::MPI::max(mpi_comm, zmax);

  std::vector<double> result;
  result.push_back((xmax + xmin)/2.0);
  result.push_back((ymax + zmin)/2.0);
  result.push_back((zmax + zmin)/2.0);

  double d = std::max(xmax - xmin, ymax - ymin);
  d = 2.0*std::max(d, zmax - zmin) + zmax ;
  result.push_back(d);

  return result;
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
                            std::vector<double>& geometry, const Mesh& mesh,
                            bool surface)
{
  // Get vertex indices
  const std::set<int> vertex_indices = surface_vertex_indices(mesh);

  std::size_t offset = dolfin::MPI::global_offset(mesh.mpi_comm(),
                                                  vertex_indices.size(), true);
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // Collect up topology of the local part of the mesh which should be
  // displayed
  std::vector<int> local_output;
  if (!surface)
  {
    for (EdgeIterator e(mesh); !e.end(); ++e)
    {
      // If one of the faces connected to this edge is external, then
      // output the edge
      bool allow_edge = (tdim == 2);
      if (!allow_edge)
      {
        for (FaceIterator f(*e); !f.end(); ++f)
        {
          if (f->num_global_entities(tdim) == 1)
            allow_edge = true;
        }
      }

      if (allow_edge)
      {
        for (VertexIterator v(*e); !v.end(); ++v)
        {
          // Find position of vertex in set
          std::size_t pos = std::distance(vertex_indices.begin(),
                                          vertex_indices.find(v->index()));
          local_output.push_back(pos + offset);
        }
        local_output.push_back(-1);
      }
    }
  }
  else
  {
    // Output faces
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (tdim == 2 or f->num_global_entities(tdim) == 1)
      {
        for (VertexIterator v(*f); !v.end(); ++v)
        {
          // Find position of vertex in set
          std::size_t pos = std::distance(vertex_indices.begin(),
                                          vertex_indices.find(v->index()));
          local_output.push_back(pos + offset);
        }
        local_output.push_back(-1);
      }
    }
  }

  // Gather up all topology on process 0
  dolfin::MPI::gather(mesh.mpi_comm(), local_output, topology);

  // Collect up geometry of all local points
  std::vector<double> local_geom_output;
  for (auto index : vertex_indices)
  {
    Vertex v(mesh, index);
    local_geom_output.push_back(v.x(0));
    local_geom_output.push_back(v.x(1));
    if (gdim == 2)
      local_geom_output.push_back(0.0);
    else
      local_geom_output.push_back(v.x(2));
  }

  // Gather up all geometry on process 0 and append to xml
  dolfin::MPI::gather(mesh.mpi_comm(), local_geom_output, geometry);
}
//-----------------------------------------------------------------------------
std::string X3DOM::array_to_string3(std::array<double, 3> x)
{
  std::string str = std::to_string(x[0]) + " " + std::to_string(x[1])
    + " " + std::to_string(x[2]);

  return str;
}
//-----------------------------------------------------------------------------
