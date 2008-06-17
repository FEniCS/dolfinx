// Copyright (C) 2005-2007 Garth N.Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Nuno Lopes 2008.
//
// First added:  2008-05-29

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/Function.h>
#include <dolfin/la/Vector.h>
#include "RAWFile.h"


using namespace dolfin;

//----------------------------------------------------------------------------
RAWFile::RAWFile(const std::string filename) : GenericFile(filename)
{
  type = "RAW";
}
//----------------------------------------------------------------------------
RAWFile::~RAWFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void RAWFile::operator<<(MeshFunction<int>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void RAWFile::operator<<(MeshFunction<unsigned int>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void RAWFile::operator<<(MeshFunction<double>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void RAWFile::operator<<(Function& u)
{
  // Update raw file name and clear file
  rawNameUpdate(counter);
        
  // Write results
  ResultsWrite(u);
  
  
  // Increase the number of times we have saved the function
  counter++;
  
  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in RAW format." << endl;

}
//----------------------------------------------------------------------------
void RAWFile::ResultsWrite(Function& u) const
{
  // Open file
  FILE *fp = fopen(raw_filename.c_str(), "a");
  
 
  const uint rank = u.rank();
  if(rank > 1)
    error("Only scalar and vectors functions can be saved in Raw format.");

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
  

  if ( dim > 3 )
    warning("Cannot handle RAW file with number of components > 3. Writing first three components only");

  fprintf(fp,"%d \n",mesh.numVertices());
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {    
    if ( rank == 0 ) 
      fprintf(fp," %e ", values[ vertex->index() ] );
    else if ( u.dim(0) == 2 ) 
      fprintf(fp," %e %e", values[ vertex->index() ], 
                                values[ vertex->index() + mesh.numVertices() ] );
    else  
      fprintf(fp," %e  %e  %e", values[ vertex->index() ], 
                               values[ vertex->index() +   mesh.numVertices() ], 
                               values[ vertex->index() + 2*mesh.numVertices() ] );

    fprintf(fp,"\n");
  }	 
  
  // Close file
  fclose(fp);

  delete [] values;
}
//----------------------------------------------------------------------------
void RAWFile::rawNameUpdate(const int counter) 
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;
  
  fileid.fill('0');
  fileid.width(6);
  
  filestart.assign(filename, 0, filename.find("."));
  extension.assign(filename, filename.find("."), filename.size());
  
  fileid << counter;
  newfilename << filestart << fileid.str() << ".raw";
  
  raw_filename = newfilename.str();
  
  // Make sure file is empty
  FILE* fp = fopen(raw_filename.c_str(), "w");
  fclose(fp);
}
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
template<class T>
void RAWFile::MeshFunctionWrite(T& meshfunction) 
{
  // Update raw file name and clear file
  rawNameUpdate(counter);

 
  Mesh& mesh = meshfunction.mesh(); 

  if( meshfunction.dim() != mesh.topology().dim() )
    error("RAW output of mesh functions is implemenetd for cell-based functions only.");    

  
  // Open file
  std::ofstream fp(raw_filename.c_str(), std::ios_base::app);
  
  fp<<mesh.numCells( ) <<std::endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    fp << meshfunction.get( cell->index() )  << std::endl;
  
  // Close file
  fp.close();

 
  // Increase the number of times we have saved the mesh function
  counter++;

  cout << "saved mesh function " << counter << " times." << endl;

  cout << "Saved mesh function " << mesh.name() << " (" << mesh.label()
       << ") to file " << filename << " in RAW format." << endl;
}    

