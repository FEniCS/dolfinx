// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2005-2006.
// Modified by Kristian Oelgaard 2006.
//
// First added:  2005-07-05
// Last changed: 2007-05-16

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/Vector.h>
#include "VTKFile.h"


using namespace dolfin;

//----------------------------------------------------------------------------
VTKFile::VTKFile(const std::string filename) : GenericFile(filename)
{
  type = "VTK";
}
//----------------------------------------------------------------------------
VTKFile::~VTKFile()
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(Mesh& mesh)
{
  // Update vtu file name and clear file
  vtuNameUpdate(counter);

  // Write pvd file
  pvdFileWrite(counter);

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
void VTKFile::operator<<(MeshFunction<int>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(MeshFunction<unsigned int>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(MeshFunction<double>& meshfunction)
{
  MeshFunctionWrite(meshfunction);
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(Function& u)
{
  // Update vtu file name and clear file
  vtuNameUpdate(counter);
  
  // Write pvd file
  pvdFileWrite(counter);
    
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
void VTKFile::MeshWrite(Mesh& mesh) const
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
void VTKFile::ResultsWrite(Function& u) const
{
  // Type of data (point or cell). Point by default.
  std::string data_type = "point";

  // Check that we have a Function that can be handled
  error("Need to add test for function type in Function.");
  //if(u.type() == Function::empty || u.type() == Function::ufc)
  //  error("Function type cannot be written in VTK format.");
 
  // Get rank of Function
  const uint rank = u.element().value_rank();
  if(rank > 1)
    error("Only scalar and vectors functions can be saved in VTK format.");

  // Get number of components
  const uint dim = u.element().value_dimension(0);
  if ( dim > 3 )
    warning("Cannot handle VTK file with number of components > 3. Writing first three components only");

  // Test for DiscreteFunction finite element type by signature
  error("Need to add test for function type");
/*
  if(u.type() == Function::discrete)
  {
    if(rank == 0)
    {
      // Test for P0 element
      if(u.elemen().signature().substr(0, 49) == "Discontinuous Lagrange finite element of degree 0")
        data_type = "cell";
      // Test for non-Lagrane element
      else if(u.element().signature().substr(0, 8) != "Lagrange")
        error("Only Lagrange functions or order k > 0 can be written in VTK format. You may need to project your function."); 
    }
    else
    {
      // FIXME: Add test for other rank elements 
    }
  }
*/

  // Open file
  FILE *fp = fopen(vtu_filename.c_str(), "a");
  
  // Get mesh
  Mesh& mesh = const_cast<Mesh&>(u.function_space().mesh());

  // Write function data at mesh cells
  if(data_type == "cell")
  {
    // Allocate memory for function values at vertices
    uint size = mesh.numCells();
    for (uint i = 0; i < u.element().value_rank(); i++)
      size *= u.element().value_dimension(i);
    double* values = new double[size];

    // Get function values on cells
    u.vector().get(values);

    // Write headers
    if (rank == 0)
    {
      fprintf(fp, "<CellData  Scalars=\"U\"> \n");
      fprintf(fp, "<DataArray  type=\"Float64\"  Name=\"U\"  format=\"ascii\">	 \n");
    }
    else
    {
      fprintf(fp, "<CellData  Vectors=\"U\"> \n");
      fprintf(fp, "<DataArray  type=\"Float64\"  Name=\"U\"  NumberOfComponents=\"3\" format=\"ascii\">	 \n");	
    }

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {    
      if ( rank == 0 ) 
        fprintf(fp," %e ", values[ cell->index() ] );
      else if ( u.element().value_dimension(0) == 2 ) 
        fprintf(fp," %e %e  0.0", values[ cell->index() ], 
                                  values[ cell->index() + mesh.numCells() ] );
      else  
        fprintf(fp," %e %e  %e", values[ cell->index() ], 
                                 values[ cell->index() +   mesh.numCells() ], 
                                 values[ cell->index() + 2*mesh.numCells() ] );
  
      fprintf(fp,"\n");
    }	 
    fprintf(fp, "</DataArray> \n");
    fprintf(fp, "</CellData> \n");

    delete [] values;
  }
  else if(data_type == "point") 
  {
    // Allocate memory for function values at vertices
    uint size = mesh.numVertices();
    for (uint i = 0; i < u.element().value_rank(); i++)
      size *= u.element().value_dimension(i);
    double* values = new double[size];

    // Get function values at vertices
    u.interpolate(values);

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

    delete [] values;
  }
  else
    error("Unknown VTK data type."); 
  
  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void VTKFile::pvdFileWrite(uint num)
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
  fname.assign(vtu_filename, filename.find_last_of("/") + 1, vtu_filename.size()); 
  
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
void VTKFile::VTKHeaderOpen(Mesh& mesh) const
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
void VTKFile::VTKHeaderClose() const
{
  // Open file
  FILE *fp = fopen(vtu_filename.c_str(), "a");
  
  // Close headers
  fprintf(fp, "</Piece> \n </UnstructuredGrid> \n </VTKFile>"); 	
  
  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void VTKFile::vtuNameUpdate(const int counter) 
{
  std::string filestart, extension;
  std::ostringstream fileid, newfilename;
  
  fileid.fill('0');
  fileid.width(6);
  
  filestart.assign(filename, 0, filename.find("."));
  extension.assign(filename, filename.find("."), filename.size());
  
  fileid << counter;
  newfilename << filestart << fileid.str() << ".vtu";
  
  vtu_filename = newfilename.str();
  
  // Make sure file is empty
  FILE* fp = fopen(vtu_filename.c_str(), "w");
  fclose(fp);
}
//----------------------------------------------------------------------------
template<class T>
void VTKFile::MeshFunctionWrite(T& meshfunction) 
{
  // Update vtu file name and clear file
  vtuNameUpdate(counter);

  // Write pvd file
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
//----------------------------------------------------------------------------

