// Copyright (C) 2005-2007 Garth N.Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes 2008.
//
// First added:  2008-07-02

#include <sstream>
#include <fstream>
#include <boost/scoped_array.hpp>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/Vector.h>
#include "XYZFile.h"

using namespace dolfin;

//----------------------------------------------------------------------------
XYZFile::XYZFile(const std::string filename) : GenericFile(filename)
{
  type = "XYZ";
}
//----------------------------------------------------------------------------
XYZFile::~XYZFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void XYZFile::operator<<(const Function& u)
{
  // Update xyz file name and clear file
  xyzNameUpdate(counter);

  // Write results
  ResultsWrite(u);

  // Increase the number of times we have saved the function
  counter++;

  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in xd3d xyz format." << endl;
}
//----------------------------------------------------------------------------
void XYZFile::ResultsWrite(const Function& u) const
{
  // Open file
  std::ofstream fp(xyz_filename.c_str(), std::ios_base::app);
  if (!fp)
    error("Unable to open file %s", filename.c_str());

  const uint rank = u.function_space().element().value_rank();
  if(rank > 1)
    error("Only scalar functions can be saved in xyz format.");

  // Get number of components
  uint dim = 1;
  for (uint i = 0; i < rank; i++)
    dim *= u.function_space().element().value_dimension(i);

  const Mesh& mesh = u.function_space().mesh();

  // Allocate memory for function values at vertices
  const uint size = mesh.num_vertices()*dim;
  boost::scoped_array<double> values(new double[size]);

  // Get function values at vertices
  u.compute_vertex_values(values.get(), mesh);

  // Write function data at mesh vertices
  if ( dim > 1 )
    error("Cannot handle XYZ file for non-scalar functions. ");
  std::ostringstream ss;
  ss << std::scientific;
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
        ss.str("");
        ss<<vertex->x(0)<<" "<< vertex->x(1)<<" "<< values[ vertex->index() ];
        ss<<std::endl;
        fp<<ss.str( );
  }
}
//----------------------------------------------------------------------------
void XYZFile::xyzNameUpdate(const int counter)
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;

  fileid.fill('0');
  fileid.width(6);

  filestart.assign(filename, 0, filename.find("."));
  extension.assign(filename, filename.find("."), filename.size());

  fileid << counter;
  newfilename << filestart << fileid.str() << ".xyz";

  xyz_filename = newfilename.str();

  // Make sure file is empty
  FILE* fp = fopen(xyz_filename.c_str(), "w");
  if (!fp)
    error("Unable to open file %s", filename.c_str());
  fclose(fp);
}
//----------------------------------------------------------------------------
template<class T>
void XYZFile::MeshFunctionWrite(T& meshfunction)
{
  // Update xyz file name and clear file
  xyzNameUpdate(counter);

  Mesh& mesh = meshfunction.mesh();

  if( meshfunction.dim() != mesh.topology().dim() )
    error("XYZ output of mesh functions is implemenetd for cell-based functions only.");

  // Open file
  std::ofstream fp(xyz_filename.c_str(), std::ios_base::app);

  fp<<mesh.num_cells( ) <<std::endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    fp << meshfunction.get( cell->index() )  << std::endl;

  // Close file
  fp.close();

  // Increase the number of times we have saved the mesh function
  counter++;

  cout << "saved mesh function " << counter << " times." << endl;

  cout << "Saved mesh function " << mesh.name() << " (" << mesh.label()
       << ") to file " << filename << " in XYZ format." << endl;
}
//----------------------------------------------------------------------------
