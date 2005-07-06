// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-07-05
// Last changed: 2005-07-05

#include <dolfin/Mesh.h>
#include <dolfin/Function.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/VTKFile.h>

using namespace dolfin;

//-­---------------------------------------------------------------------------
VTKFile::VTKFile(const std::string filename) : GenericFile(filename)
{
  type = "VTK";
}
//-­---------------------------------------------------------------------------
VTKFile::~VTKFile()
{
  // Do nothing
}
//-­---------------------------------------------------------------------------
void VTKFile::operator<<(Mesh& mesh)
{

  dolfin_info("Saving mesh to VTK file.");
  
  // Write headers
  VTKHeaderOpen(mesh);

  MeshWrite(mesh);
  
  // Write headers
  VTKHeaderClose();
  
}
//-­---------------------------------------------------------------------------
void VTKFile::operator<<(Function& u)
{

  dolfin_info("Writing Function to VTK file");
  
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
  ++u;
  
  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in VTK format." << endl;
}
//-­---------------------------------------------------------------------------
void VTKFile::MeshWrite(const Mesh& mesh) const
{

  // Open file
  FILE* fp = fopen(filename.c_str(), "a");

  // Write node positions
  fprintf(fp, "<Points>  \n");
  fprintf(fp, "<DataArray  type=\"Float32\"  NumberOfComponents=\"3\"  format=\"ascii\">  \n");
  for (NodeIterator n(mesh); !n.end(); ++n)
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
    for (NodeIterator n(c); !n.end(); ++n) fprintf(fp," %8d ",n->id());
    fprintf(fp," \n");
  }  
  fprintf(fp, "</DataArray> \n");

  // Write offset into connectivity array for the end of each cell
  fprintf(fp, "<DataArray  type=\"Int32\"  Name=\"offsets\"  format=\"ascii\">  \n");
  for (int offsets = 1; offsets <= mesh.noCells(); offsets++)
  {
    if (mesh.type() == Mesh::tetrahedra ) fprintf(fp, " %8d \n",  offsets*4);
    if (mesh.type() == Mesh::triangles )    fprintf(fp, " %8d \n", offsets*3);
  }
  fprintf(fp, "</DataArray> \n");
  
  //Write cell type
  fprintf(fp, "<DataArray  type=\"UInt8\"  Name=\"types\"  format=\"ascii\">  \n");
  for (int types = 1; types <= mesh.noCells(); types++)
  {
    if (mesh.type() == Mesh::tetrahedra ) fprintf(fp, " 10 \n");
    if (mesh.type() == Mesh::triangles )    fprintf(fp, " 5 \n");
  }
  fprintf(fp, "</DataArray> \n");
  fprintf(fp, "</Cells> \n"); 
  
  // Close file
  fclose(fp);

}
//-­---------------------------------------------------------------------------
void VTKFile::ResultsWrite(Function& u) const
{

  uint no_components = 0;
  const FiniteElement& element = u.element();

  if ( element.rank() == 0 )
  {
    no_components = 1;
  }
  else if ( element.rank() == 1 )
  {
    no_components = element.tensordim(0);
  }
  else
    dolfin_error("Cannot handle tensor valued functions.");

  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  //Write PointData displacement	
  if(no_components == 1)
  {
    fprintf(fp, "<PointData  Scalars=\"U\"> \n");
    fprintf(fp, "<DataArray  type=\"Float32\"  Name=\"U\"  format=\"ascii\">	 \n");
  }
  else
  {
    fprintf(fp, "<PointData  Vectors=\"U\"> \n");
    fprintf(fp, "<DataArray  type=\"Float32\"  Name=\"U\"  NumberOfComponents=\"3\" format=\"ascii\">	 \n");	
  }
  
  for (NodeIterator n(u.mesh()); !n.end(); ++n)
  {    
    for(uint i =0; i < no_components; ++i)
    {
      fprintf(fp," %e ",u(*n, i));
    }
    fprintf(fp,"\n");
  }	 
  fprintf(fp, "</DataArray> \n");
  fprintf(fp, "</PointData> \n");
  
  
  // Close file
  fclose(fp);

}
//-­---------------------------------------------------------------------------
void VTKFile::VTKHeaderOpen(const Mesh& mesh) const
{

  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write headers
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\"  version=\"0.1\"  byte_order=\"LittleEndian\"  compressor=\"vtkZLibDataCompressor\">  \n");
  fprintf(fp, "<UnstructuredGrid>  \n");
  fprintf(fp, "<Piece  NumberOfPoints=\" %8d\"  NumberOfCells=\" %8d\">  \n", mesh.noNodes(), mesh.noCells());
  
  // Close file
  fclose(fp);
  
}
//-­---------------------------------------------------------------------------
void VTKFile::VTKHeaderClose() const
{
  
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Close headers
  fprintf(fp, "</Piece> \n </UnstructuredGrid> \n </VTKFile>"); 	
  
  // Close file
  fclose(fp);
  

}
//-­---------------------------------------------------------------------------
