// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <dolfin/Grid.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/OpenDXFile.h>

using namespace dolfin;

//-­---------------------------------------------------------------------------
OpenDXFile::OpenDXFile(const std::string filename) : GenericFile(filename)
{

}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator>>(Vector& x)
{
  dolfin_error("Cannot read vectors from OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator>>(Matrix& A)
{
  dolfin_error("Cannot read matrices from OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator>>(Grid& grid)
{
  dolfin_error("Cannot read grids from OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator>>(Function& u)
{
  dolfin_error("Cannot read functions from OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Vector& x)
{
  dolfin_error("Cannot save vectors to OpenDX files.");  
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Matrix& A)
{
  dolfin_error("Cannot save matrices to OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Grid& grid)
{
  dolfin_info("Saving grid to OpenDX file.");

  int noNodes = grid.noNodes();
  int noCells = grid.noCells();

  // Open file
  FILE* fp = fopen(filename.c_str(), "a");

  // Write nodes
  fprintf(fp, "# A list of all node positions\n");
  fprintf(fp, "object \"nodes\" class array type float rank 1 shape 3 items %d lsb binary data follows\n", noNodes);

  for (NodeIterator n(grid); !n.end(); ++n) {

    Point p = n->coord();
    
    double x = p.x;
    double y = p.y;
    double z = p.z;
    
    fwrite(&x, sizeof(double), 1, fp);
    fwrite(&y, sizeof(double), 1, fp);
    fwrite(&z, sizeof(double), 1, fp);
    
  }
  fprintf(fp,"\n\n");

  // Write elements
  fprintf(fp, "# A list of all elements (connections)\n");
  fprintf(fp, "object \"cells\" class array type int rank 1 shape 4 items %d lsb binary data follows\n", noCells);
  
  for (NodeIterator n(grid); !n.end(); ++n) {
    for (CellIterator c(n); !c.end(); ++c) {
      int id  = c->id();
      fwrite(&id, sizeof(int), 1, fp);
    }
  }
  fprintf(fp, "\n");
  fprintf(fp, "attribute \"element type\" string \"tetrahedra\"\n");
  fprintf(fp, "attribute \"ref\" string \"positions\"\n");
  fprintf(fp, "\n\n");  

  // Write the grid
  fprintf(fp, "object \"grid\" class field\n");
  fprintf(fp,"component \"positions\" value \"nodes\"\n");
  fprintf(fp,"component \"connections\" value \"cells\"\n");

  // Close file
  fclose(fp);
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Function& u)
{
  dolfin_error("Cannot save functions to OpenDX files.");
}
//-­---------------------------------------------------------------------------
