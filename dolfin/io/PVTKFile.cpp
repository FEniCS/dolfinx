// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2005-2006.
// Modified by Kristian Oelgaard 2006.
// Modified by Niclas Jansson 2008.
//
// First added:  2005-07-05
// Last changed: 2008-06-26

#include <dolfin/fem/FiniteElement.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/Vector.h>
#include "PVTKFile.h"


using namespace dolfin;

//----------------------------------------------------------------------------
PVTKFile::PVTKFile(const std::string filename) : GenericFile(filename)
{
  type = "VTK";
}
//----------------------------------------------------------------------------
PVTKFile::~PVTKFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(Mesh& mesh)
{
  // Update vtu file name and clear file
  vtuNameUpdate(counter);

  // Only the root updates the pvd file
  if(MPI::process_number() == 0) {    

    // Update pvtu file name and clear file
    pvtuNameUpdate(counter);

    // Write pvd file
    pvdFileWrite(counter);
    
    // Write pvtu file
    pvtuFileWrite();
  }

  // Write headers
  VTKHeaderOpen(mesh);

  // Write mesh
  MeshWrite(mesh);
  
  // Close headers
  VTKHeaderClose();

  // Increase the number of times we have saved the mesh
  counter++;

  message(1, "Saved mesh %s (%s) to file %s in VTK format.",
          mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(MeshFunction<int>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(MeshFunction<unsigned int>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(MeshFunction<double>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void PVTKFile::operator<<(Function& u)
{
  // Update vtu file name and clear file
  vtuNameUpdate(counter);

  // Write pvd file

  // Only the root updates the pvd file
  if(MPI::process_number() == 0) {
    
    // Update pvtu file name and clear file
    pvtuNameUpdate(counter);

    // Write pvd file
    pvdFileWrite(counter);

    // Write pvtu file
    pvtuFileWrite_func(u);
  }
    
  // FIXME: need to fix const-ness
  Mesh& mesh = const_cast<Mesh&>(u.function_space().mesh()); 

  // Write headers
  VTKHeaderOpen(mesh);
  
  // Write Mesh
  MeshWrite(mesh);
  
  // Write results
  ResultsWrite(u);
  
  // Close headers
  VTKHeaderClose();
  
  // Increase the number of times we have saved the function
  counter++;
  
  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in VTK format." << endl;

}
//----------------------------------------------------------------------------
void PVTKFile::MeshWrite(Mesh& mesh) const
{
  // Open file
  FILE* fp = fopen(vtu_filename.c_str(), "a");

  // Write vertex positions
  fprintf(fp, "<Points>  \n");
  fprintf(fp, "<DataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\"ascii\">  \n");
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    Point p = v->point();
    fprintf(fp," %f %f %f \n", p.x(), p.y(), p.z());
  }
  fprintf(fp, "</DataArray>  \n");
  fprintf(fp, "</Points>  \n");
  
  // Write cell connectivity
  fprintf(fp, "<Cells>  \n");
  fprintf(fp, "<DataArray  type=\"Int32\"  Name=\"connectivity\"  format=\"ascii\">  \n");
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    for (VertexIterator v(*c); !v.end(); ++v)
      fprintf(fp," %8u ",v->index());
    fprintf(fp," \n");
  }  
  fprintf(fp, "</DataArray> \n");

  // Write offset into connectivity array for the end of each cell
  fprintf(fp, "<DataArray  type=\"Int32\"  Name=\"offsets\"  format=\"ascii\">  \n");
  for (uint offsets = 1; offsets <= mesh.numCells(); offsets++)
  {
    if (mesh.type().cellType() == CellType::tetrahedron )
      fprintf(fp, " %8u \n",  offsets*4);
    if (mesh.type().cellType() == CellType::triangle )
      fprintf(fp, " %8u \n", offsets*3);
    if (mesh.type().cellType() == CellType::interval )
      fprintf(fp, " %8u \n",  offsets*2);
  }
  fprintf(fp, "</DataArray> \n");
  
  //Write cell type
  fprintf(fp, "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\">  \n");
  for (uint types = 1; types <= mesh.numCells(); types++)
  {
    if (mesh.type().cellType() == CellType::tetrahedron )
      fprintf(fp, " 10 \n");
    if (mesh.type().cellType() == CellType::triangle )
      fprintf(fp, " 5 \n");
    if (mesh.type().cellType() == CellType::interval )
      fprintf(fp, " 3 \n");
  }
  fprintf(fp, "</DataArray> \n");
  fprintf(fp, "</Cells> \n"); 
  
  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void PVTKFile::ResultsWrite(Function& u) const
{
  // Open file
  FILE *fp = fopen(vtu_filename.c_str(), "a");
  
  Mesh& mesh = const_cast<Mesh&>(u.function_space().mesh());

  const uint rank = u.element().value_rank();
  if(rank > 1)
    error("Only scalar and vectors functions can be saved in VTK format.");

  // Get number of components
  const uint dim = u.element().value_dimension(0);

  // Allocate memory for function values at vertices
  uint size = mesh.numVertices();
  for (uint i = 0; i < u.element().value_rank(); i++)
    size *= u.element().value_dimension(i);
  double* values = new double[size];

  // Get function values at vertices
  u.interpolate(values);

  // Write function data at mesh vertices
  if ( rank == 0 )
  {
    fprintf(fp, "<PointData  Scalars=\"U\"> \n");
    fprintf(fp, "<DataArray  type=\"Float64\"  Name=\"U\"  format=\"ascii\">	 \n");
  }
  else
  {
    fprintf(fp, "<PointData  Vectors=\"U\"> \n");
    fprintf(fp, "<DataArray  type=\"Float64\"  Name=\"U\"  NumberOfComponents=\"3\" format=\"ascii\">	 \n");	
  }

  if ( dim > 3 )
    warning("Cannot handle VTK file with number of components > 3. Writing first three components only");
	
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {    
    if ( rank == 0 ) 
      fprintf(fp," %e ", values[ vertex->index() ] );
    else if ( u.element().value_dimension(0) == 2 ) 
      fprintf(fp," %e %e  0.0", values[ vertex->index() ], 
                                values[ vertex->index() + mesh.numVertices() ] );
    else  
      fprintf(fp," %e %e  %e", values[ vertex->index() ], 
                               values[ vertex->index() +   mesh.numVertices() ], 
                               values[ vertex->index() + 2*mesh.numVertices() ] );

    fprintf(fp,"\n");
  }	 
  fprintf(fp, "</DataArray> \n");
  fprintf(fp, "</PointData> \n");
  
  // Close file
  fclose(fp);

  delete [] values;
}
//----------------------------------------------------------------------------
void PVTKFile::pvdFileWrite(uint num)
{
  std::fstream pvdFile;

  if( num == 0)
  {
    // Open pvd file
    pvdFile.open(filename.c_str(), std::ios::out|std::ios::trunc);
    // Write header    
    pvdFile << "<?xml version=\"1.0\"?> " << std::endl;
    pvdFile << "<VTKFile type=\"Collection\" version=\"0.1\" > " << std::endl;
    pvdFile << "<Collection> " << std::endl;
  } 
  else
  {
    // Open pvd file
    pvdFile.open(filename.c_str(),  std::ios::out|std::ios::in);
    pvdFile.seekp(mark);
  
  }
  // Remove directory path from name for pvd file
  std::string fname;
  fname.assign(pvtu_filename, filename.find_last_of("/") + 1, pvtu_filename.size()); 
  
  // Data file name 
  pvdFile << "<DataSet timestep=\"" << num << "\" part=\"0\"" << " file=\"" <<  fname <<  "\"/>" << std::endl; 
  mark = pvdFile.tellp();
  
  // Close headers
  pvdFile << "</Collection> " << std::endl;
  pvdFile << "</VTKFile> " << std::endl;
  
  // Close file
  pvdFile.close();  

}
//----------------------------------------------------------------------------
void PVTKFile::pvtuFileWrite()
{
  std::fstream pvtuFile;

  
  // Open pvtu file
  pvtuFile.open(pvtu_filename.c_str(), std::ios::out|std::ios::trunc);
  // Write header
  pvtuFile << "<?xml version=\"1.0\"?> " << std::endl;
  pvtuFile << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\">" << std::endl;
  pvtuFile << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
  
  pvtuFile << "<PCellData>" << std::endl;
  pvtuFile << "<PDataArray  type=\"Int32\"  Name=\"connectivity\"  format=\"ascii\"/>" << std::endl;
  pvtuFile << "<PDataArray  type=\"Int32\"  Name=\"offsets\"  format=\"ascii\"/>" << std::endl;
  pvtuFile << "<PDataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\"/>"  << std::endl;
  pvtuFile<<"</PCellData>" << std::endl;
  
  pvtuFile << "<PPoints>" <<std::endl;
  pvtuFile << "<PDataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\"ascii\"/>" << std::endl;
  pvtuFile << "</PPoints>" << std::endl;

  // Remove rank from vtu filename ( <rank>.vtu)
  std::string fname;
  fname.assign(vtu_filename, filename.find_last_of("/") + 1, vtu_filename.size() - 5 ); 
  for(uint i=0; i< MPI::num_processes(); i++)
    pvtuFile << "<Piece Source=\"" << fname << i << ".vtu\"/>" << std::endl; 
    
  pvtuFile << "</PUnstructuredGrid>" << std::endl;
  pvtuFile << "</VTKFile>" << std::endl;
  pvtuFile.close();
    
}//----------------------------------------------------------------------------
void PVTKFile::pvtuFileWrite_func(Function& u)
{
  std::fstream pvtuFile;

  
  // Open pvtu file
  pvtuFile.open(pvtu_filename.c_str(), std::ios::out|std::ios::trunc);
  // Write header
  pvtuFile << "<?xml version=\"1.0\"?> " << std::endl;
  pvtuFile << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\">" << std::endl;
  pvtuFile << "<PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
  
  if(u.element().value_rank() == 0) {
    pvtuFile << "<PPointData Scalars=\"U\">" << std::endl;    
    pvtuFile << "<PDataArray  type=\"Float64\"  Name=\"U\"  format=\"ascii\"/>" << std::endl;
  }
  else {
    pvtuFile << "<PPointData Vectors=\"U\">" << std::endl;    
    pvtuFile << "<PDataArray  type=\"Float64\"  Name=\"U\"  NumberOfComponents=\"3\" format=\"ascii\"/>" << std::endl;
    }
  
  pvtuFile << "</PPointData>" << std::endl;


  pvtuFile << "<PCellData>" << std::endl;
  pvtuFile << "<PDataArray  type=\"Int32\"  Name=\"connectivity\"  format=\"ascii\"/>" << std::endl;
  pvtuFile << "<PDataArray  type=\"Int32\"  Name=\"offsets\"  format=\"ascii\"/>" << std::endl;
  pvtuFile << "<PDataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\"/>"  << std::endl;
  pvtuFile<<"</PCellData>" << std::endl;
  
  pvtuFile << "<PPoints>" <<std::endl;
  pvtuFile << "<PDataArray  type=\"Float64\"  NumberOfComponents=\"3\"  format=\"ascii\"/>" << std::endl;
  pvtuFile << "</PPoints>" << std::endl;

  std::string fname;
  // Remove rank from vtu filename ( <rank>.vtu)
  fname.assign(vtu_filename, filename.find_last_of("/") + 1, vtu_filename.size() - 5 ); 
  for(uint i=0; i< MPI::num_processes(); i++)
    pvtuFile << "<Piece Source=\"" << fname << i << ".vtu\"/>" << std::endl; 
  
  
  pvtuFile << "</PUnstructuredGrid>" << std::endl;
  pvtuFile << "</VTKFile>" << std::endl;
  pvtuFile.close();
    
}
//----------------------------------------------------------------------------
void PVTKFile::VTKHeaderOpen(Mesh& mesh) const
{
  // Open file
  FILE *fp = fopen(vtu_filename.c_str(), "a");
  
  // Write headers
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\"  version=\"0.1\"   >\n");
  fprintf(fp, "<UnstructuredGrid>  \n");
  fprintf(fp, "<Piece  NumberOfPoints=\" %8u\"  NumberOfCells=\" %8u\">  \n",
	  mesh.numVertices(), mesh.numCells());
  
  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void PVTKFile::VTKHeaderClose() const
{
  // Open file
  FILE *fp = fopen(vtu_filename.c_str(), "a");
  
  // Close headers
  fprintf(fp, "</Piece> \n </UnstructuredGrid> \n </VTKFile>"); 	
  
  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void PVTKFile::vtuNameUpdate(const int counter) 
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;
  
  fileid.fill('0');
  fileid.width(6);
  
  filestart.assign(filename, 0, filename.find("."));
  extension.assign(filename, filename.find("."), filename.size());
  
  fileid << counter;
   newfilename << filestart << fileid.str() << "_" << MPI::process_number() <<".vtu";
  vtu_filename = newfilename.str();
  
  // Make sure file is empty
  FILE* fp = fopen(vtu_filename.c_str(), "w");
  fclose(fp);
}
//----------------------------------------------------------------------------
void PVTKFile::pvtuNameUpdate(const int counter)
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;
  
  fileid.fill('0');
  fileid.width(6);
  
  filestart.assign(filename, 0, filename.find("."));
  extension.assign(filename, filename.find("."), filename.size());
  
  fileid << counter;
  newfilename << filestart << fileid.str() << ".pvtu";
  
  pvtu_filename = newfilename.str();
  
  // Make sure file is empty
  FILE* fp = fopen(pvtu_filename.c_str(), "w");
  fclose(fp);
}
//----------------------------------------------------------------------------
template<class T>
void PVTKFile::MeshFunctionWrite(T& meshfunction) 
{
  // Update vtu file name and clear file
  vtuNameUpdate(counter);

  // Write pvd file
  if(MPI::process_number() == 0) 
    pvdFileWrite(counter);

  Mesh& mesh = meshfunction.mesh(); 

  if( meshfunction.dim() != mesh.topology().dim() )
    error("VTK output of mesh functions is implemenetd for cell-based functions only.");    

  // Write headers
  VTKHeaderOpen(mesh);

  // Write mesh
  MeshWrite(mesh);
  
  // Open file
  std::ofstream fp(vtu_filename.c_str(), std::ios_base::app);

  fp << "<CellData  Scalars=\"U\">" << std::endl;
  fp << "<DataArray  type=\"Float64\"  Name=\"U\"  format=\"ascii\">" << std::endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    fp << meshfunction.get( cell->index() )  << std::endl;
  fp << "</DataArray>" << std::endl;
  fp << "</CellData>" << std::endl;
  
  // Close file
  fp.close();

  // Close headers
  VTKHeaderClose();

  // Increase the number of times we have saved the mesh function
  counter++;

  cout << "saved mesh function " << counter << " times." << endl;

  cout << "Saved mesh function " << mesh.name() << " (" << mesh.label()
       << ") to file " << filename << " in VTK format." << endl;
}    
//-----------------------------------------------------------------------------

