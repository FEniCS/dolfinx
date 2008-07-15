// Copyright (C) 2005-2007 Garth N.Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes 2008.
//
// First added:  2008-07-02

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/Function.h>
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
void XYZFile::operator<<(Function& u)
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
void XYZFile::ResultsWrite(Function& u) const
{
  // Open file
  FILE *fp = fopen(xyz_filename.c_str(), "a");
  
 
  const uint rank = u.rank();
  if(rank > 1)
    error("Only scalar functions can be saved in xyz format.");

  // Get number of components
  const uint dim = u.dim(0);

  Mesh& mesh = u.mesh();
  
  // Allocate memory for function values at vertices
  uint size = mesh.numVertices();
  for (uint i = 0; i < u.rank(); i++)
    size *= u.dim(i);
  real* values = new real[size];

  // Get function values at vertices
  u.interpolate(values);

  
  // Write function data at mesh vertices
  

  if ( dim > 1 )
    error("Cannot handle XYZ file for non-scalar functions. ");

  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {    
    if ( rank == 0 ) 
      fprintf(fp,"%e %e  %e",vertex->x(0),vertex->x(1), values[ vertex->index() ] );
    fprintf(fp,"\n");
  }	 
  
  // Close file
  fclose(fp);

  delete [] values;
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
  
  fp<<mesh.numCells( ) <<std::endl;
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
