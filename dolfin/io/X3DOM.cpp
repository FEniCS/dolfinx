#include <iostream>
#include <sstream>
#include <fstream>
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
std::string X3DOM::str(const Mesh& mesh, FacetType facet_type)
{
  // Create empty pugi XML doc
  pugi::xml_document xml_doc;

  // Build XML
  x3dom_xml(xml_doc, mesh, facet_type);

  // Convert XML doc to string
  std::stringstream s;
  const std::string indent = "  ";
  xml_doc.save(s, indent.c_str());
  return s.str();
}
//-----------------------------------------------------------------------------
void X3DOM::x3dom_xml(pugi::xml_node& xml_doc, const Mesh& mesh,
                      FacetType facet_type)
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

  // Add standard boilerplate XML for X3D, adjusting field of view to
  // the size of the object, given by xpos.
  add_xml_header(xml_doc, xpos, facet_type);

  // Compute set of vertices that lie on boundary
  const std::set<int> surface_vertices = surface_vertex_indices(mesh);

  add_mesh_to_xml(xml_doc, mesh, surface_vertices, facet_type);
}
//-----------------------------------------------------------------------------
std::string X3DOM::html(const Mesh& mesh, FacetType facet_type)
{
  // Create empty pugi XML doc
  pugi::xml_document xml_doc;

  // Add html node
  pugi::xml_node node = xml_doc.append_child("html");

  // Add head node
  pugi::xml_node head = node.append_child("head");

  // Add script
  pugi::xml_node script = head.append_child("script");
  script.append_attribute("type") = "text/javascript";
  script.append_attribute("src") = "http://www.x3dom.org/download/x3dom.js";
  script.append_child(pugi::node_pcdata);//.set_value();

  // Add link
  pugi::xml_node link = head.append_child("link");
  link.append_attribute("rel") = "stylesheet";
  link.append_attribute("type") = "text/css";
  link.append_attribute("href") = "http://www.x3dom.org/download/x3dom.css";

  // Add body node
  pugi::xml_node body = node.append_child("body");

  // Add X3D XML
  x3dom_xml(body, mesh, facet_type);

  // Convert XML doc to string, without default XML header
  std::stringstream s;
  const std::string indent = "  ";
  xml_doc.save(s, indent.c_str(), pugi::format_default | pugi::format_no_declaration);
  return s.str();
}
//-----------------------------------------------------------------------------
/*
std::string X3DOM::str(const MeshFunction<std::size_t>& meshfunction,
                       const std::string facet_type, const size_t palette)
{
  // Get mesh
  dolfin_assert(meshfunction.mesh());
  const Mesh& mesh = *meshfunction.mesh();

  // Ensure connectivity has been computed
  mesh.init(mesh.topology().dim() - 1 , mesh.topology().dim());

  // Mesh geometric dimension
  const std::size_t gdim = mesh.geometry().dim();

  // Mesh topological dimension
  const std::size_t tdim = mesh.topology().dim();

  // MeshFunction dimension
  const std::size_t cell_dim = meshfunction.dim();

  // Check that MeshFunction dimension is handled
  if (cell_dim != tdim)
  {
    dolfin_error("X3DFile.cpp",
                 "output meshfunction",
                 "Can only output CellFunction at present");
  }

  // Check that X3D type is appropriate
  if (cell_dim == tdim && facet_type == "IndexedLineSet")
  {
    dolfin_error("X3DFile.cpp",
                 "output meshfunction",
                 "Cannot output CellFunction with Edge mesh");
  }

  // Check that mesh is in 2D or 3D
  if (gdim !=2 && gdim !=3)
  {
    dolfin_error("X3DFile.cpp",
                 "output mesh",
                 "X3D will only output 2D or 3D meshes");
  }

  // Pointer to MeshFunction data
  const std::size_t* values = meshfunction.values();

  // Get min/max values of MeshFunction
  std::size_t minval = *std::min_element(values, values + meshfunction.size());
  minval = MPI::min(mesh.mpi_comm(), minval);
  std::size_t maxval = *std::max_element(values, values + meshfunction.size());
  maxval = MPI::max(mesh.mpi_comm(), maxval);
  double dval;
  if (maxval == minval)
    dval = 1.0;
  else
    dval = 255.0/(double)(maxval - minval);

  // Get mesh min/max  dimensions and viewpoint
  const std::vector<double> xpos = mesh_min_max(mesh);

  // Get MPI details
  const std::size_t rank = dolfin::MPI::rank(mesh.mpi_comm());

  // Create pugi xml document
  pugi::xml_document xml_doc;

  // Write XML header
  if (MPI::rank(mesh.mpi_comm()) == 0)
    add_xml_header(xml_doc, xpos, facet_type);

  // Make a set of the indices we wish to use. In 3D, we are ignoring
  // all interior facets, so reducing the number of vertices
  // substantially
  const std::vector<std::size_t> vecindex = vertex_index(mesh);

  // Write vertices
  add_mesh_to_xml(xml_doc, mesh, vecindex, facet_type);

  // Iterate over mesh facets
  std::vector<unsigned int> local_output;
  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    // Check if topological dimension is 2, or if we have a boundary
    // facet in 3D
    if (tdim == 2 || f->num_global_entities(tdim) == 1)
    {
      // Get cell connected to facet
      CellIterator cell(*f);

      // Write mesh function value to string stream
      local_output.push_back((unsigned int)
                             ((meshfunction[*cell] - minval)*dval));
    }
  }

  // Gather up data on zero
  std::vector<unsigned int> gathered_output;
  MPI::gather(mesh.mpi_comm(), local_output, gathered_output);

  // Export string on root process
  if (rank == 0)
  {
    pugi::xml_node indexed_face_set = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type_to_x3d_str(facet_type));
    indexed_face_set.append_attribute("colorPerVertex") = "false";

    std::stringstream str_output;
    for (std::vector<unsigned int>::iterator val = gathered_output.begin();
         val != gathered_output.end(); ++val)
    {
      str_output << *val << " ";
    }

    indexed_face_set.append_attribute("colorIndex") = str_output.str().c_str();

    // Output palette
    pugi::xml_node color = indexed_face_set.append_child("Color");
    color.append_attribute("color") = color_palette(palette).c_str();
    // xml_doc.save_file(_filename.c_str(), "  ");
  }
  // Output string
  std::stringstream ss;
  xml_doc.save(ss, "  ");
  return ss.str();
}
*/
//-----------------------------------------------------------------------------
/*
std::string X3DOM::str(const Function& u,
                       const std::string facet_type, const size_t palette)
{
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  // Mesh geometric and topological dimensions
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();

  if (gdim !=2 && gdim !=3)
  {
    dolfin_error("X3DFile.cpp",
                 "output mesh",
                 "X3D will only output 2D or 3D meshes");
  }

  // Build mesh connectivity
  mesh.init(tdim - 1 , tdim);

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  // Only allow scalar fields
  if (u.value_rank() > 1)
  {
    dolfin_error("X3DFile.cpp",
                 "write X3D",
                 "Can only handle scalar and vector Functions");
  }

  if (u.value_rank() == 1)
    warning("X3DFile outputs scalar magnitude of vector field");

  // Only allow vertex centered data
  const bool vertex_data = (dofmap.max_element_dofs() != 1);
  if (!vertex_data)
  {
    dolfin_error("X3DFile.cpp",
                 "write X3D",
                 "Can only handle vertex-based Function at present");
  }

  // Compute vertex data values
  std::vector<double> data_values;
  u.compute_vertex_values(data_values, mesh);

  // Normalise data values
  if (u.value_rank() == 1)
  {
    std::vector<double> magnitude;
    const std::size_t num_vertices = mesh.num_vertices();
    for (std::size_t i = 0; i < num_vertices ; ++i)
    {
      double val = 0.0;
      for (std::size_t j = 0; j < u.value_size() ; j++)
        val += pow(data_values[i + j*num_vertices], 2.0);
      val = sqrt(val);
      magnitude.push_back(val);
    }
    data_values.resize(magnitude.size());
    std::copy(magnitude.begin(), magnitude.end(), data_values.begin());
  }

  // Create pugi document
  pugi::xml_document xml_doc;

  // Get mesh mix/max dimensions and write XML header
  const std::vector<double> xpos = mesh_min_max(mesh);
  add_xml_header(xml_doc, xpos, facet_type);

  // Get indices of vertices on mesh surface
  const std::set<int> surface_vertices = vertex_index(mesh);
  const std::vector<std::size_t> surface_vertices_vec(surface_vertices.begin(),
                                                      surface_vertices.end());

  // Write vertices and vertex data to XML file
  add_mesh_to_xml(xml_doc, mesh, surface_vertices_vec, facet_type);
  add_values_to_xml(xml_doc, mesh, surface_vertices_vec,
                    data_values, facet_type, palette);

  // Output string
  std::stringstream ss;
  if (MPI::rank(mesh.mpi_comm()) == 0)
    xml_doc.save(ss, "  ");
  return ss.str();
}
*/
//-----------------------------------------------------------------------------
/*
std::string X3DOM::html_str(const MeshFunction<std::size_t>& meshfunction,
                            const std::string facet_type, const size_t palette)
{
  // Return html string for HTML
  std::string start_str = "<html> \n"
                          "    <head> \n"
                          "        <script type='text/javascript' src='http://www.x3dom.org/download/x3dom.js'> </script> \n"
                          "        <link rel='stylesheet' type='text/css' href='http://www.x3dom.org/download/x3dom.css'></link> \n"
                          "    </head> \n"
                          "</html> \n"
                          "\n"
                          "<body>\n";

  std::stringstream ss;
  ss << start_str << xml_str(meshfunction, facet_type, palette) << "</body>";

  return ss.str();
}
*/
//-----------------------------------------------------------------------------
/*
void X3DOM::xml_to_file(const std::string filename, const Mesh& mesh,
                        const std::string facet_type, const size_t palette)
{
  // Save XML string to file
  // Check if extension is X3D and give warning
  if(filename.substr(filename.find_last_of(".") + 1) == "x3d") {
    std::ofstream out(filename);
    out << xml_str(mesh, facet_type, palette);
    out.close();
  }
  else {
    dolfin_error("X3DOM.cpp",
             "output file type",
             "File type should be *.x3d");
  }
}
*/
//-----------------------------------------------------------------------------
/*
void X3DOM::html_to_file(const std::string filename, const Mesh& mesh,
                         const std::string facet_type, const size_t palette)
{
  // Save HTMl string to file
  // Check if extension is X3D and give warning
  if(filename.substr(filename.find_last_of(".") + 1) == "html") {
    std::ofstream out(filename);
    out << html_str(mesh, facet_type, palette);
    out.close();
  }
  else {
    dolfin_error("X3DOM.cpp",
             "output file type",
             "File type should be *.html");
  }
}
*/
//-----------------------------------------------------------------------------
void X3DOM::add_values_to_xml(pugi::xml_node& xml_doc, const Mesh& mesh,
                              const std::vector<std::size_t>& vecindex,
                              const std::vector<double>& data_values,
                              FacetType facet_type, const std::size_t palette)
{
  const std::size_t tdim = mesh.topology().dim();

  double minval = *std::min_element(data_values.begin(), data_values.end());
  minval = MPI::min(mesh.mpi_comm(), minval);
  double maxval = *std::max_element(data_values.begin(), data_values.end());
  maxval = MPI::max(mesh.mpi_comm(), maxval);

  double scale = 0.0;
  if (maxval == minval)
    scale = 1.0;
  else
    scale = 255.0/(maxval - minval);

  std::vector<int> local_output;
  if (facet_type == FacetType::wireframe)
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
          local_output.push_back((int)((data_values[v->index()] - minval)
                                       *scale));
        }
        local_output.push_back(-1);
      }
    }
  }
  else if (facet_type == FacetType::facet)
  {
    // Output faces
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (tdim == 2 || f->num_global_entities(tdim) == 1)
      {
        for (VertexIterator v(*f); !v.end(); ++v)
        {
          local_output.push_back((int)((data_values[v->index()] - minval)
                                       *scale));
        }
        local_output.push_back(-1);
      }
    }
  }

  // Gather up on zero
  std::vector<int> gathered_output;
  dolfin::MPI::gather(mesh.mpi_comm(), local_output, gathered_output);
  if (dolfin::MPI::rank(mesh.mpi_comm()) == 0)
  {
    pugi::xml_node indexed_face_set = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type_to_x3d_str(facet_type));
    indexed_face_set.append_attribute("colorPerVertex") = "true";

    std::stringstream str_output;
    for (auto val : gathered_output)
      str_output << val << " ";
    indexed_face_set.append_attribute("colorIndex") = str_output.str().c_str();

    // Output colour palette
    pugi::xml_node color = indexed_face_set.append_child("Color");
    color.append_attribute("color") = color_palette(palette).c_str();
  }
}
//-----------------------------------------------------------------------------
void X3DOM::add_mesh_to_xml(pugi::xml_node& xml_doc, const Mesh& mesh,
                            const std::set<int>& vertex_indices,
                            FacetType facet_type)
{
  std::size_t offset = dolfin::MPI::global_offset(mesh.mpi_comm(),
                                                  vertex_indices.size(), true);

  const std::size_t rank = dolfin::MPI::rank(mesh.mpi_comm());
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // Collect up topology of the local part of the mesh which should be
  // displayed

  std::vector<int> local_output;
  if (facet_type == FacetType::wireframe)
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
          std::size_t pos = std::distance(vertex_indices.begin(), vertex_indices.find(v->index()));

          local_output.push_back(pos + offset);
        }
        local_output.push_back(-1);
      }
    }
  }
  else if (facet_type == FacetType::facet)
  {
    // Output faces
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (tdim == 2 || f->num_global_entities(tdim) == 1)
      {
        for (VertexIterator v(*f); !v.end(); ++v)
        {
          // Find position of vertex in set
          std::size_t pos = std::distance(vertex_indices.begin(), vertex_indices.find(v->index()));

          local_output.push_back(pos + offset);
        }
        local_output.push_back(-1);
      }
    }
  }

  // Gather up all topology on process 0 and append to xml
  std::vector<int> gathered_output;
  dolfin::MPI::gather(mesh.mpi_comm(), local_output, gathered_output);
  if (rank == 0)
  {
    pugi::xml_node indexed_face_set
      = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type_to_x3d_str(facet_type));

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
    pugi::xml_node indexed_face_set
      = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type_to_x3d_str(facet_type));

    pugi::xml_node coordinate = indexed_face_set.append_child("Coordinate");

    std::stringstream str_output;
    for (auto val : gathered_geom_output)
      str_output << val << " ";
    coordinate.append_attribute("point") = str_output.str().c_str();
  }
}
//-----------------------------------------------------------------------------
void X3DOM::add_xml_header(pugi::xml_node& xml_doc,
                           const std::vector<double>& xpos,
                           FacetType facet_type)
{
  xml_doc.append_child(pugi::node_doctype).set_value("X3D PUBLIC \"ISO//Web3D//DTD X3D 3.2//EN\" \"http://www.web3d.org/specifications/x3d-3.2.dtd\"");

  pugi::xml_node x3d = xml_doc.append_child("X3D");
  x3d.append_attribute("profile") = "Interchange";
  x3d.append_attribute("version") = "3.2";
  x3d.append_attribute("xmlns:xsd")
    = "http://www.w3.org/2001/XMLSchema-instance";
  x3d.append_attribute("xsd:noNamespaceSchemaLocation")
    = "http://www.web3d.org/specifications/x3d-3.2.xsd";
  x3d.append_attribute("width") = "500px";
  x3d.append_attribute("height") = "400px";

  pugi::xml_node scene = x3d.append_child("Scene");

  pugi::xml_node shape = scene.append_child("Shape");
  pugi::xml_node material
    = shape.append_child("Appearance").append_child("Material");
  material.append_attribute("ambientIntensity") = "0.4";
  material.append_attribute("shininess") = "0.8";
  material.append_attribute("diffuseColor") = "0.7 0.7 0.7";
  material.append_attribute("specularColor") = "0.2 0.2 0.2";
  material.append_attribute("emmisiveColor") = "0.7 0.7 0.7";

  shape.append_child(facet_type_to_x3d_str(facet_type)).append_attribute("solid") = "false";

  // Have to append Background after shape
  pugi::xml_node background = scene.append_child("Background");
  background.append_attribute("skyColor") = "0.319997 0.340002 0.429999";

  // Append viewpoint after shape
  pugi::xml_node viewpoint = scene.append_child("Viewpoint");
  std::string xyz = boost::lexical_cast<std::string>(xpos[0]) + " "
      + boost::lexical_cast<std::string>(xpos[1]) + " "
    + boost::lexical_cast<std::string>(xpos[3]);
  viewpoint.append_attribute("position") = xyz.c_str();

  viewpoint.append_attribute("orientation") = "0 0 0 1";
  viewpoint.append_attribute("fieldOfView") = "0.785398";
  xyz = boost::lexical_cast<std::string>(xpos[0]) + " "
    + boost::lexical_cast<std::string>(xpos[1]) + " "
    + boost::lexical_cast<std::string>(xpos[2]);
  viewpoint.append_attribute("centerOfRotation") = xyz.c_str();

  viewpoint.append_attribute("zNear") = "-1";
  viewpoint.append_attribute("zFar") = "-1";

  // Append ambient light
  pugi::xml_node ambient_light = scene.append_child("DirectionalLight");
  ambient_light.append_attribute("ambientIntensity") = "1";
  ambient_light.append_attribute("intensity") = "0";
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
    if (tdim == 2 || f->num_global_entities(tdim) == 1)
    {
      for (VertexIterator v(*f); !v.end(); ++v)
        vertex_set.insert(v->index());
    }
  }

  return vertex_set;
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
const char* X3DOM::facet_type_to_x3d_str(FacetType facet_type)
{
  // Map from enum to X3D string
  switch (facet_type)
  {
  case FacetType::facet:
    return "IndexedFaceSet";
    break;
  case FacetType::wireframe:
    return "IndexedLineSet";
    break;
  default:
    dolfin_error("X3DOM.cpp",
                 "mesh style",
                 "Unknown mesh output type");
    return "error";
  }
}
//-----------------------------------------------------------------------------