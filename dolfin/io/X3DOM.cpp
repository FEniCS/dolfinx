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

std::string X3DOM::xml_str(const Mesh& mesh, const std::string facet_type, const size_t palette)
{
  // Return xml string for X3D
  // return "This will print XML string.";

  const std::size_t gdim = mesh.geometry().dim();
  if (gdim !=2 && gdim !=3)
  {
    dolfin_error("X3DFile.cpp",
                 "output mesh",
                 "X3D will only output 2D or 3D meshes");
  }

  // Create pugi doc
  pugi::xml_document xml_doc;

  // For serial - ensure connectivity
  mesh.init(mesh.topology().dim() - 1 , mesh.topology().dim());

  // Get mesh max and min dimensions and viewpoint
  const std::vector<double> xpos = mesh_min_max(mesh);

  // Create XML for all mesh vertices on surface
  output_xml_header(xml_doc, xpos, facet_type);
  const std::vector<std::size_t> vecindex = vertex_index(mesh);
  write_vertices(xml_doc, mesh, vecindex, facet_type);

  // if (MPI::rank(mesh.mpi_comm()) == 0)
  //   xml_doc.save_file(_filename.c_str(), "  ");

  std::stringstream ss;
  xml_doc.save(ss, "  ");
  
  return ss.str();
}
//-----------------------------------------------------------------------------
std::string X3DOM::xml_str(const MeshFunction<std::size_t>& meshfunction, const std::string facet_type, const size_t palette)
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
  const std::size_t process_number = MPI::rank(mesh.mpi_comm());

  // Create pugi xml document
  pugi::xml_document xml_doc;

  // Write XML header
  if (MPI::rank(mesh.mpi_comm()) == 0)
    output_xml_header(xml_doc, xpos, facet_type);

  // Make a set of the indices we wish to use. In 3D, we are ignoring
  // all interior facets, so reducing the number of vertices
  // substantially
  const std::vector<std::size_t> vecindex = vertex_index(mesh);

  // Write vertices
  write_vertices(xml_doc, mesh, vecindex, facet_type);

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
  if (process_number == 0)
  {
    pugi::xml_node indexed_face_set = xml_doc.child("X3D")
      .child("Scene").child("Shape").child(facet_type.c_str());
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
//-----------------------------------------------------------------------------
std::string X3DOM::html_str(const Mesh& mesh, const std::string facet_type, const size_t palette)
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
  ss << start_str << xml_str(mesh, facet_type, palette) << "</body>";

  return ss.str();
}   
//-----------------------------------------------------------------------------
std::string X3DOM::html_str(const MeshFunction<std::size_t>& meshfunction, const std::string facet_type, const size_t palette)
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
//-----------------------------------------------------------------------------
void X3DOM::xml_to_file(const std::string filename, const Mesh& mesh, const std::string facet_type, const size_t palette)
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
//-----------------------------------------------------------------------------
void X3DOM::html_to_file(const std::string filename, const Mesh& mesh, const std::string facet_type, const size_t palette)
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
//-----------------------------------------------------------------------------
void X3DOM::write_values(pugi::xml_document& xml_doc, const Mesh& mesh,
                           const std::vector<std::size_t> vecindex,
                           const std::vector<double> data_values, const std::string facet_type, const std::size_t palette)
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
  if (facet_type == "IndexedLineSet")
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
  else if (facet_type == "IndexedFaceSet")
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
  MPI::gather(mesh.mpi_comm(), local_output, gathered_output);
  if (MPI::rank(mesh.mpi_comm()) == 0)
  {
    pugi::xml_node indexed_face_set = xml_doc.child("X3D")
      .child("Scene").child("Shape").child(facet_type.c_str());
    indexed_face_set.append_attribute("colorPerVertex") = "true";

    std::stringstream str_output;
    for (std::vector<int>::iterator val = gathered_output.begin();
         val != gathered_output.end(); ++val)
    {
      str_output << *val << " ";
    }
    indexed_face_set.append_attribute("colorIndex") = str_output.str().c_str();

    // Output colour palette
    pugi::xml_node color = indexed_face_set.append_child("Color");
    color.append_attribute("color") = color_palette(palette).c_str();
  }
}
//-----------------------------------------------------------------------------
void X3DOM::write_vertices(pugi::xml_document& xml_doc, const Mesh& mesh,
                             const std::vector<std::size_t> vecindex, const std::string facet_type)
{
  std::size_t offset = MPI::global_offset(mesh.mpi_comm(), vecindex.size(),
                                          true);

  const std::size_t process_number = MPI::rank(mesh.mpi_comm());
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  std::vector<int> local_output;
  if (facet_type == "IndexedLineSet")
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
          std::size_t index_it  = std::find(vecindex.begin(),
                                            vecindex.end(),
                                            v->index()) - vecindex.begin();
          local_output.push_back(index_it + offset);
        }
        local_output.push_back(-1);
      }
    }
  }
  else if (facet_type == "IndexedFaceSet")
  {
    // Output faces
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (tdim == 2 || f->num_global_entities(tdim) == 1)
      {
        for (VertexIterator v(*f); !v.end(); ++v)
        {
          std::size_t index_it = std::find(vecindex.begin(),
                                           vecindex.end(),
                                           v->index()) - vecindex.begin();
          local_output.push_back(index_it + offset);
        }
        local_output.push_back(-1);
      }
    }
  }

  std::vector<int> gathered_output;
  MPI::gather(mesh.mpi_comm(), local_output, gathered_output);

  if (process_number == 0)
  {
    pugi::xml_node indexed_face_set
      = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type.c_str());

    std::stringstream str_output;
    for (std::vector<int>::iterator val = gathered_output.begin();
         val != gathered_output.end(); ++val)
    {
      str_output << *val << " ";
    }
    indexed_face_set.append_attribute("coordIndex") = str_output.str().c_str();
  }

  // Now fill in the geometry
  std::vector<double> local_geom_output;
  for (std::vector<std::size_t>::const_iterator index = vecindex.begin();
       index != vecindex.end(); ++index)
  {
    Vertex v(mesh, *index);
    local_geom_output.push_back(v.x(0));
    local_geom_output.push_back(v.x(1));
    if (gdim == 2) 
      local_geom_output.push_back(0.0);
    else
      local_geom_output.push_back(v.x(2));
  }

  std::vector<double> gathered_geom_output;
  MPI::gather(mesh.mpi_comm(), local_geom_output, gathered_geom_output);

  // Finally, close off with the XML footer on process zero
  if (process_number == 0)
  {
    pugi::xml_node indexed_face_set
      = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type.c_str());

    pugi::xml_node coordinate = indexed_face_set.append_child("Coordinate");

    std::stringstream str_output;
    for (std::vector<double>::iterator val = gathered_geom_output.begin();
         val != gathered_geom_output.end(); ++val)
    {
      str_output << *val << " ";
    }
    coordinate.append_attribute("point") = str_output.str().c_str();
  }
}
//-----------------------------------------------------------------------------
void X3DOM::output_xml_header(pugi::xml_document& xml_doc,
                                const std::vector<double>& xpos, const std::string facet_type)
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
  material.append_attribute("ambientIntensity") = "0.05";
  material.append_attribute("shininess") = "0.5";
  material.append_attribute("diffuseColor") = "0.7 0.7 0.7";
  material.append_attribute("specularColor") = "0.9 0.9 0.9";
  material.append_attribute("emmisiveColor") = "0.7 0.7 0.7";

  shape.append_child(facet_type.c_str()).append_attribute("solid") = "false";

  // Have to append Background after shape
  pugi::xml_node background = scene.append_child("Background");
  background.append_attribute("skyColor") = "0.9 0.9 1.0";

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

  xmin = MPI::min(mpi_comm, xmin);
  ymin = MPI::min(mpi_comm, ymin);
  zmin = MPI::min(mpi_comm, zmin);

  xmax = MPI::max(mpi_comm, xmax);
  ymax = MPI::max(mpi_comm, ymax);
  zmax = MPI::max(mpi_comm, zmax);

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
std::vector<std::size_t> X3DOM::vertex_index(const Mesh& mesh) 
{
  const std::size_t tdim = mesh.topology().dim();
  std::set<std::size_t> vindex;

  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    // If in 3D, only output exterior faces
    // FIXME: num_global_entities not working in serial
    if (tdim == 2 || f->num_global_entities(tdim) == 1)
    {
      for (VertexIterator v(*f); !v.end(); ++v)
        vindex.insert(v->index());
    }
  }

  // Copy to vector for wider iterator support
  const std::vector<std::size_t> vecindex(vindex.begin(), vindex.end());
  return vecindex;
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
