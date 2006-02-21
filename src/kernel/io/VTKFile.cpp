// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005-2006.
//
// First added:  2005-07-05
// Last changed: 2005-12-20

#include <dolfin/Mesh.h>
#include <dolfin/Function.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/VTKFile.h>

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
  //dolfin_info("Saving mesh to VTK file.");
  
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

  cout << "saved mesh " << mesh.number() << " times." << endl;

  cout << "Saved mesh " << mesh.name() << " (" << mesh.label()
       << ") to file " << filename << " in VTK format." << endl;
}
//----------------------------------------------------------------------------
void VTKFile::operator<<(Function& u)
{
  //dolfin_info("Writing Function to VTK file.");

  // Update vtu file name and clear file
  vtuNameUpdate(counter);
  
  // Write pvd file
  pvdFileWrite(counter);
    
  const Mesh& mesh = u.mesh(); 

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
void VTKFile::MeshWrite(const Mesh& mesh) const
{
  // Open file
  FILE* fp = fopen(vtu_filename.c_str(), "a");

  // Write vertex positions
  fprintf(fp, "<Points>  \n");
  fprintf(fp, "<DataArray  type=\"Float32\"  NumberOfComponents=\"3\"  format=\"ascii\">  \n");
  for (VertexIterator n(mesh); !n.end(); ++n)
  {
    Point   p = n->coord();
    fprintf(fp," %f %f %f \n", p.x, p.y, p.z);
  }
  fprintf(fp, "</DataArray>  \n");
  fprintf(fp, "</Points>  \n");
  
  // Write cell connectivity
  fprintf(fp, "<Cells>  \n");
  fprintf(fp, "<DataArray  type=\"Int32\"  Name=\"connectivity\"  format=\"ascii\">  \n");
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    for (VertexIterator n(c); !n.end(); ++n) fprintf(fp," %8d ",n->id());
    fprintf(fp," \n");
  }  
  fprintf(fp, "</DataArray> \n");

  // Write offset into connectivity array for the end of each cell
  fprintf(fp, "<DataArray  type=\"Int32\"  Name=\"offsets\"  format=\"ascii\">  \n");
  for (int offsets = 1; offsets <= mesh.numCells(); offsets++)
  {
    if (mesh.type() == Mesh::tetrahedra )   fprintf(fp, " %8d \n",  offsets*4);
    if (mesh.type() == Mesh::triangles )    fprintf(fp, " %8d \n", offsets*3);
  }
  fprintf(fp, "</DataArray> \n");
  
  //Write cell type
  fprintf(fp, "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\">  \n");
  for (int types = 1; types <= mesh.numCells(); types++)
  {
    if (mesh.type() == Mesh::tetrahedra )   fprintf(fp, " 10 \n");
    if (mesh.type() == Mesh::triangles )    fprintf(fp, " 5 \n");
  }
  fprintf(fp, "</DataArray> \n");
  fprintf(fp, "</Cells> \n"); 
  
  // Close file
  fclose(fp);

}
//----------------------------------------------------------------------------
void VTKFile::ResultsWrite(Function& u) const
{
  // Open file
  FILE *fp = fopen(vtu_filename.c_str(), "a");
  
  //Write PointData
  if ( u.vectordim() == 1 )
  {
    fprintf(fp, "<PointData  Scalars=\"U\"> \n");
    fprintf(fp, "<DataArray  type=\"Float32\"  Name=\"U\"  format=\"ascii\">	 \n");
  }
  else
  {
    fprintf(fp, "<PointData  Vectors=\"U\"> \n");
    fprintf(fp, "<DataArray  type=\"Float32\"  Name=\"U\"  NumberOfComponents=\"3\" format=\"ascii\">	 \n");	
  }

  if ( u.vectordim() > 3 )
    dolfin_warning("Cannot handle VTK file with number of components > 3. Writing first three components only");
	
  for (VertexIterator n(u.mesh()); !n.end(); ++n)
  {    
    if ( u.vectordim() == 1 ) 
    {
      fprintf(fp," %e ",u(*n, 0));
    }
    else if ( u.vectordim() == 2 ) 
    {
      fprintf(fp," %e %e  0.0",u(*n, 0), u(*n, 1));
    }
    else  
    {
      fprintf(fp," %e %e  %e",u(*n, 0), u(*n, 1), u(*n, 2));
    }
    fprintf(fp,"\n");
  }	 
  fprintf(fp, "</DataArray> \n");
  fprintf(fp, "</PointData> \n");
  
  
  // Close file
  fclose(fp);
}
//----------------------------------------------------------------------------
void VTKFile::pvdFileWrite(int num)
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
void VTKFile::VTKHeaderOpen(const Mesh& mesh) const
{
  // Open file
  FILE *fp = fopen(vtu_filename.c_str(), "a");
  
  // Write headers
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\"  version=\"0.1\"   >\n");
  fprintf(fp, "<UnstructuredGrid>  \n");
  fprintf(fp, "<Piece  NumberOfPoints=\" %8d\"  NumberOfCells=\" %8d\">  \n",
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
