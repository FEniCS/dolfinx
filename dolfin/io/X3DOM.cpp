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

#include <sstream>
#include <boost/lexical_cast.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Vertex.h>

#include "X3DOM.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
/*
std::string X3DOM::str(const Mesh& mesh)
{
  X3DParameters default_parameters;
  return str(mesh, default_parameters);
}
*/
//-----------------------------------------------------------------------------
std::string X3DOM::str(const Mesh& mesh, X3DParameters parameters)
{
  // Check material vector
  //if (check_colour(material_colour, bg))
  {
    // Create empty pugi XML doc
    pugi::xml_document xml_doc;

    // Build X3D XML and add to XML doc
    x3dom_xml(xml_doc, mesh, parameters.get_representation(),
              parameters.get_viewpoint_buttons(),
              parameters.get_background_colour());

    // Save XML doc to stringstream
    std::stringstream s;
    const std::string indent = "  ";
    xml_doc.save(s, indent.c_str());

    // Return string
    return s.str();
  }
  //else
  //  return std::string();
}
//-----------------------------------------------------------------------------
/*
std::string X3DOM::html(const Mesh& mesh)
{
  X3DParameters default_parameters;
  return html(mesh, default_parameters);
}
*/
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
  x3dom_xml(body_node, mesh, parameters.get_representation(),
            parameters.get_viewpoint_buttons(),
            parameters.get_background_colour());

  // FIXME Write Function to do this
  // Now append four viewpoint buttons to 'body' node
  if (parameters.get_viewpoint_buttons())
  {
    // Add viewpoint control node
    pugi::xml_node viewpoint_control = body_node.append_child("div");

    // Add attributes to viewpoint niode
    viewpoint_control.append_attribute("id") = "camera_buttons";
    viewpoint_control.append_attribute("style") = "display: block";

    add_viewpoint_control_option(viewpoint_control, "Front");
    add_viewpoint_control_option(viewpoint_control, "Back");
    add_viewpoint_control_option(viewpoint_control, "Left");
    add_viewpoint_control_option(viewpoint_control, "Right");
    add_viewpoint_control_option(viewpoint_control, "Top");
    add_viewpoint_control_option(viewpoint_control, "Bottom");
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
pugi::xml_node X3DOM::add_x3d(pugi::xml_node& xml_node)
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
void X3DOM::x3dom_xml(pugi::xml_node& xml_node, const Mesh& mesh,
                      X3DParameters::Representation representation,
                      bool show_viewpoint_buttons,
                      const std::array<double, 3> bg)
{
  // Check that mesh is embedded in 2D or 3D
  const std::size_t gdim = mesh.geometry().dim();
  if (gdim !=2 and gdim !=3)
  {
    dolfin_error("X3DOM.cpp",
                 "get X3DOM string representation of a mesh",
                 "X3D works only for 2D and 3D meshes");
  }

  // Intialise facet-to-cell connectivity
  const std::size_t tdim = mesh.geometry().dim();
  mesh.init(tdim - 1 , tdim);

  // Get mesh max and min dimensions, needed to calculate field of
  // view
  const std::vector<double> xpos = mesh_min_max(mesh);

  // X3D doctype
  add_doctype(xml_node);

  // Add X3D node
  pugi::xml_node x3d_node = add_x3d(xml_node);
  dolfin_assert(x3d_node);

  // Add boilerplate XML to X3D node, adjusting field of view to the
  // size of the object, given by xpos
  pugi::xml_node scene = add_xml_header(x3d_node, xpos, representation,
                                        show_viewpoint_buttons, bg);
  dolfin_assert(scene);

  // Add mesh to 'shape' XML node, based on shape id
  if (representation == X3DParameters::Representation::surface_with_edges)
  {
    // First add the facet
    pugi::xml_node shape = scene.find_child_by_attribute("Shape", "id", x3d_str(X3DParameters::Representation::surface).c_str());
    add_mesh(shape, mesh, X3DParameters::Representation::surface);

    // Then the edge
    shape = scene.find_child_by_attribute("Shape", "id", x3d_str(X3DParameters::Representation::wireframe).c_str());
    add_mesh(shape, mesh, X3DParameters::Representation::wireframe);
  }
  else
  {
    pugi::xml_node shape = scene.find_child_by_attribute("Shape", "id", x3d_str(representation).c_str());
    add_mesh(shape, mesh, representation);
  }

  // Append text for mesh info
  pugi::xml_node mesh_info = x3d_node.append_child("div");
  dolfin_assert(mesh_info);
  mesh_info.append_attribute("style") = "position: absolute; bottom: 2%; left: 2%; text-align: left; font-size: 12px; color: white;";
  std::string data = "Number of vertices: " + std::to_string(mesh.num_vertices())
    + ", number of cells: " + std::to_string(mesh.num_cells());
  mesh_info.append_child(pugi::node_pcdata).set_value(data.c_str());
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
void X3DOM::add_mesh(pugi::xml_node& xml_node, const Mesh& mesh,
                     X3DParameters::Representation representation)
{
  // Get vertex indices
  const std::set<int> vertex_indices = surface_vertex_indices(mesh);

  std::size_t offset = dolfin::MPI::global_offset(mesh.mpi_comm(),
                                                  vertex_indices.size(), true);
  const std::size_t rank = dolfin::MPI::rank(mesh.mpi_comm());
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // Collect up topology of the local part of the mesh which should be
  // displayed
  std::vector<int> local_output;
  if (representation == X3DParameters::Representation::wireframe)
  {
    for (EdgeIterator e(mesh); !e.end(); ++e)
    {
      bool allow_edge = (tdim == 2);

      // If one of the faces connected to this edge is external, then
      // output the edge
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
  else if (representation == X3DParameters::Representation::surface)
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

  // Gather up all topology on process 0 and append to xml
  std::vector<int> gathered_output;
  dolfin::MPI::gather(mesh.mpi_comm(), local_output, gathered_output);

  // Add topology data to XML tree
  pugi::xml_node indexed_face_set;
  if (rank == 0)
  {
    // Add edges node
    indexed_face_set = xml_node.append_child(x3d_str(representation).c_str());
    indexed_face_set.append_attribute("solid") = "false";

    // Add data to edges node
    std::stringstream str_output;
    for (auto val : gathered_output)
      str_output << val << " ";
    indexed_face_set.append_attribute("coordIndex") = str_output.str().c_str();
  }

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
  std::vector<double> gathered_geom_output;
  dolfin::MPI::gather(mesh.mpi_comm(), local_geom_output, gathered_geom_output);
  if (rank == 0)
  {
    // Add coordinate node
    pugi::xml_node coordinate = indexed_face_set.append_child("Coordinate");

    // Add data to coordinate node
    std::stringstream str_output;
    for (auto val : gathered_geom_output)
      str_output << val << " ";
    coordinate.append_attribute("point") = str_output.str().c_str();
  }
}
//-----------------------------------------------------------------------------
pugi::xml_node
X3DOM::add_xml_header(pugi::xml_node& x3d_node,
                      const std::vector<double>& xpos,
                      X3DParameters::Representation representation,
                      bool show_viewpoint_buttons,
                      const std::array<double, 3> bg)
{
  pugi::xml_node scene = x3d_node.append_child("Scene");

  if (representation == X3DParameters::Representation::surface_with_edges)
  {
    // Append edge mesh first so the facet will be on top after being
    // appended later
    add_shape_node(scene, X3DParameters::Representation::wireframe);
    add_shape_node(scene, X3DParameters::Representation::surface);
  }
  else
    add_shape_node(scene, representation);

  // Have to append Background after shape
  pugi::xml_node background = scene.append_child("Background");
  std::string background_str = boost::lexical_cast<std::string>(bg[0]) + " "
    + boost::lexical_cast<std::string>(bg[1]) + " "
    + boost::lexical_cast<std::string>(bg[2]);
  background.append_attribute("skyColor") = background_str.c_str();

  // Append viewpoint after shape
  add_viewpoint_xml_nodes(scene, xpos, show_viewpoint_buttons);

  // Append ambient light
  pugi::xml_node ambient_light = scene.append_child("DirectionalLight");
  ambient_light.append_attribute("ambientIntensity") = "1";
  ambient_light.append_attribute("intensity") = "0";

  return scene;
}
//-----------------------------------------------------------------------------
void X3DOM::add_viewpoint_control_option(pugi::xml_node& viewpoint_control,
                                         std::string vp)
{
  std::string onclick_str
    = "document.getElementById('" + vp + "').setAttribute('set_bind','true');";
  pugi::xml_node viewpoint_buttons = viewpoint_control.append_child("button");
  viewpoint_buttons.append_attribute("onclick") = onclick_str.c_str();
  viewpoint_buttons.append_attribute("style") = "display: block";
  viewpoint_buttons.append_child(pugi::node_pcdata).set_value(vp.c_str());
}
//-----------------------------------------------------------------------------
void X3DOM::add_viewpoint_xml_nodes(pugi::xml_node& xml_scene,
                                    const std::vector<double>& xpos,
                                    bool show_viewpoint_buttons)
{
  // Center of rotation
  std::string center_of_rotation
    = boost::lexical_cast<std::string>(xpos[0]) + " "
    + boost::lexical_cast<std::string>(xpos[1]) + " "
    + boost::lexical_cast<std::string>(xpos[2]);

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
  add_viewpoint_node(xml_scene, Viewpoint::default_view, center_of_rotation, xpos);
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
    position = boost::lexical_cast<std::string>(xpos[0]) + " "
      + boost::lexical_cast<std::string>(xpos[1] + xpos[3] - xpos[2]) + " "
      + boost::lexical_cast<std::string>(xpos[2]);
    break;
  case Viewpoint::bottom:
    viewpoint_str = "bottom";
    orientation = "1 0 0 1.5707963267948";
    position = boost::lexical_cast<std::string>(xpos[0]) + " "
      + boost::lexical_cast<std::string>(xpos[1] - xpos[3] + xpos[2]) + " "
      + boost::lexical_cast<std::string>(xpos[2]);
    break;
  case Viewpoint::left:
    viewpoint_str = "left";
    orientation = "0 1 0 1.5707963267948";
    position = boost::lexical_cast<std::string>(xpos[0] + xpos[3] - xpos[2]) + " "
      + boost::lexical_cast<std::string>(xpos[1]) + " "
      + boost::lexical_cast<std::string>(xpos[2]);;
    break;
  case Viewpoint::right:
    viewpoint_str = "right";
    orientation = "0 -1 0 1.5707963267948";
    position = boost::lexical_cast<std::string>(xpos[0] - xpos[3] + xpos[2]) + " "
      + boost::lexical_cast<std::string>(xpos[1]) + " "
      + boost::lexical_cast<std::string>(xpos[2]);
    break;
  case Viewpoint::back:
    viewpoint_str = "back";
    orientation = "0 1 0 3.1415926535898";
    position = boost::lexical_cast<std::string>(xpos[0]) + " "
      + boost::lexical_cast<std::string>(xpos[1]) + " "
      + boost::lexical_cast<std::string>(xpos[2] - xpos[3]);
    break;
  case Viewpoint::front:
    viewpoint_str = "front";
    orientation = "0 0 0 1";
    position = boost::lexical_cast<std::string>(xpos[0]) + " "
      + boost::lexical_cast<std::string>(xpos[1]) + " "
      + boost::lexical_cast<std::string>(xpos[3]);
    break;
  case Viewpoint::default_view:
    viewpoint_str = "default";
    orientation = "-0.7071067812 0.7071067812 0 1";
    position = boost::lexical_cast<std::string>(xpos[0] + 0.7071067812*(xpos[3] - xpos[2])) + " "
      + boost::lexical_cast<std::string>(xpos[1] + 0.7071067812*(xpos[3] - xpos[2])) + " "
      + boost::lexical_cast<std::string>(xpos[2] + 0.7071067812*(xpos[3] - xpos[2]));
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
void X3DOM::add_shape_node(pugi::xml_node& x3d_scene,
                           X3DParameters::Representation representation)
{
  pugi::xml_node shape = x3d_scene.prepend_child("Shape");
  shape.append_attribute("id") = x3d_str(representation).c_str();

  /*
  pugi::xml_node appearance = shape.append_child("Appearance");

  // Getting the string for these colour properties
  // Diffusive Colour
  std::string diffusive_col =
    boost::lexical_cast<std::string>(mat_col[0]) + " "
    + boost::lexical_cast<std::string>(mat_col[1]) + " "
    + boost::lexical_cast<std::string>(mat_col[2]);

  // Emmissive colour
  std::string emmissive_col =
    boost::lexical_cast<std::string>(mat_col[3]) + " "
    + boost::lexical_cast<std::string>(mat_col[4]) + " "
    + boost::lexical_cast<std::string>(mat_col[5]);

  // Specular colour
  std::string specular_col =
    boost::lexical_cast<std::string>(mat_col[6]) + " "
    + boost::lexical_cast<std::string>(mat_col[7]) + " "
    + boost::lexical_cast<std::string>(mat_col[8]);

  // Ambient intensity
  std::string ambient = boost::lexical_cast<std::string>(mat_col[9]);

  // Shininess
  std::string shininess = boost::lexical_cast<std::string>(mat_col[10]);

  // Transparency
  std::string transparency = boost::lexical_cast<std::string>(mat_col[11]);

  // Now append these attributes
  pugi::xml_node material = appearance.append_child("Material");
  material.append_attribute("diffuseColor") = diffusive_col.c_str();
  material.append_attribute("emmisiveColor") = emmissive_col.c_str();
  material.append_attribute("specularColor") = specular_col.c_str();
  material.append_attribute("ambientIntensity") = ambient.c_str();
  material.append_attribute("shininess") = shininess.c_str();
  material.append_attribute("transparency") = transparency.c_str();
  */
}
//-----------------------------------------------------------------------------
std::string X3DOM::color_palette(const size_t palette)
{
  // Make a basic palette of 256 colours
  std::stringstream colour;
  switch (palette)
  {
  case 1:
    for (int i = 0; i < 256; ++i)
    {
      const double x = (double)i/255.0;
      const double y = 1.0 - x;
      const double r = 4*pow(x, 3) - 3*pow(x, 4);
      const double g = 4*pow(x, 2)*(1.0 - pow(x, 2));
      const double b = 4*pow(y, 3) - 3*pow(y, 4);
      colour << r << " " << g << " " << b << " ";
    }
    break;
  case 2:
    for (int i = 0; i < 256; ++i)
    {
      const double lm = 425.0 + 250.0*(double)i/255.0;
      const double b
        = 1.8*exp(-pow((lm - 450.0)/((lm>450.0) ? 40.0 : 20.0), 2.0));
      const double g
        = 0.9*exp(-pow((lm - 550.0)/((lm>550.0) ? 60 : 40.0), 2.0));
      double r = 1.0*exp(-pow((lm - 600.0)/((lm>600.0) ? 40.0 : 50.0), 2.0));
      r += 0.3*exp(-pow((lm - 450.0)/((lm>450.0) ? 20.0 : 30.0), 2.0));
      const double tot = (r + g + b);

      colour << r/tot << " " << g/tot << " " << b/tot << " ";
    }
    break;
  default:
    for (int i = 0; i < 256 ; ++i)
    {
      const double r = (double)i/255.0;
      const double g = (double)(i*(255 - i))/(128.0*127.0);
      const double b = (double)(255 - i)/255.0;
      colour << r << " " << g << " " << b << " ";
    }
    break;
  }

  return colour.str();
}
//-----------------------------------------------------------------------------
std::string X3DOM::x3d_str(X3DParameters::Representation representation)
{
  // Map from enum to X3D string
  switch (representation)
  {
  case X3DParameters::Representation::surface:
    return "IndexedFaceSet";
  case X3DParameters::Representation::wireframe:
    return "IndexedLineSet";
  default:
    dolfin_error("X3DOM.cpp",
                 "mesh style",
                 "Unknown mesh output type");

    // Return string to keep compiler happy
    return "error";
  }
}
//-----------------------------------------------------------------------------
