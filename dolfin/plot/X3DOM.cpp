#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>

#include "pugixml.hpp"

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
X3DOM::X3DOM(const Mesh& mesh) 
{
  // On constructing store the mesh into xml_doc
  // TODO: Add similar constructors for MeshFunction and Function

  const std::size_t gdim = mesh.geometry().dim();
  if (gdim !=2 && gdim !=3)
  {
    dolfin_error("X3DOM.cpp",
                 "output mesh",
                 "X3DOM will only work with 2D or 3D meshes");
  }

  // For serial - ensure connectivity
  mesh.init(mesh.topology().dim() - 1 , mesh.topology().dim());

  // Get mesh max and min dimensions and viewpoint
  const std::vector<double> xpos = mesh_min_max(mesh);

  // Create XML for all mesh vertices on surface
  output_xml_header(xml_doc, xpos);
  const std::vector<std::size_t> vecindex = vertex_index(mesh);
  write_vertices(xml_doc, mesh, vecindex);
}
//-----------------------------------------------------------------------------
X3DOM::~X3DOM()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
// Print out XML string
// const std::string xml()
// {

// }

// Print out HTML string
// const std::string html()
// {

// }

// Save the XML for X3DOM in a file
// void save(std::string filename) const;
// {

// }

//-----------------------------------------------------------------------------
// Write mesh function
// void X3DFile::write_mesh(const Mesh& mesh)
// {
//   const std::size_t gdim = mesh.geometry().dim();
//   if (gdim !=2 && gdim !=3)
//   {
//     dolfin_error("X3DOM.cpp",
//                  "output mesh",
//                  "X3D will only output 2D or 3D meshes");
//   }

//   // For serial - ensure connectivity
//   mesh.init(mesh.topology().dim() - 1 , mesh.topology().dim());

//   // Get mesh max and min dimensions and viewpoint
//   const std::vector<double> xpos = mesh_min_max(mesh);

//   // Create XML for all mesh vertices on surface
//   output_xml_header(xml_doc, xpos);
//   const std::vector<std::size_t> vecindex = vertex_index(mesh);
//   write_vertices(xml_doc, mesh, vecindex);

//   if (MPI::rank(mesh.mpi_comm()) == 0)
//     xml_doc.save_file(_filename.c_str(), "  ");
// }