// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// New output format for Matrix added by Erik Svensson 2003

// FIXME: Use streams rather than stdio
#include <stdio.h>

#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Grid.h>
#include <dolfin/Function.h>

#include "MatlabFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MatlabFile::operator>>(Vector& x)
{
  // FIXME: Use logging system
  std::cout << "Warning: Cannot read vectors from Matlab files." << std::endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator>>(Matrix& A)
{
  // FIXME: Use logging system
  std::cout << "Warning: Cannot read matrices from Matlab files." << std::endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator>>(Grid& grid)
{
  // FIXME: Use logging system
  std::cout << "Warning: Cannot read grids from Matlab files." << std::endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator>>(Function& u)
{
  // FIXME: Use logging system
  std::cout << "Warning: Cannot read functions from Matlab files." << std::endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(const Vector& x)
{
  writeVector(x, filename, x.name(), x.label());
  std::cout << "Saved vector " << x.name() << " (" << x.label()
				<< ") to file " << filename << " in Matlab format." << std::endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(const Matrix& A)
{
  writeMatrix(A, filename, A.name(), A.label());
  std::cout << "Saved matrix " << A.name() << " (" << A.label()
				<< ") to file " << filename << " in Matlab format." << std::endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(const Grid& grid)
{
  writeGrid(grid, filename, grid.name(), grid.label());
  std::cout << "Saved grid " << grid.name() << " (" << grid.label()
				<< ") to file " << filename << " in Matlab format." << std::endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::operator<<(const Function& u)
{
  writeFunction(u, filename, u.name(), u.label());
  std::cout << "Saved function " << u.name() << " (" << u.label()
				<< ") to file " << filename << " in Matlab format." << std::endl;
}
//-----------------------------------------------------------------------------
void MatlabFile::writeVector(const Vector& x,
									  const std::string& filename,
									  const std::string& name,
									  const std::string& label)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write vector
  fprintf(fp, "%s = [", name.c_str());
  for (int i = 0; i < x.size(); i++)
	 fprintf(fp, " %.16e", x(i));
  fprintf(fp, " ];\n");

  // Close file
  fclose(fp);
}
//-----------------------------------------------------------------------------
void MatlabFile::writeMatrix(const Matrix& A,
									  const std::string& filename,
									  const std::string& name,
									  const std::string& label)
{
  real value;
  int j ;

  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Write matrix in sparse format
  fprintf(fp, "%s = [", name.c_str());
  for (int i = 0; i < A.size(0); i++) {
    for (int pos = 0; !A.endrow(i, pos); pos++) {
      value = A(i, &j, pos);
		fprintf(fp, " %i %i %.16e", i + 1, j + 1, value);		
		if ( i == (A.size(0) - 1) && A.endrow(i, pos + 1) )
        fprintf(fp, "];\n");
		else {
		  fprintf(fp, "\n");
		}
    }
  }
  fprintf(fp, "%s = spconvert(%s);\n", name.c_str(), name.c_str());

  // Close file
  fclose(fp);
}
//-----------------------------------------------------------------------------
void MatlabFile::writeGrid(const Grid& grid,
									const std::string& filename,
									const std::string& name,
									const std::string& label)
{
  Point p;
  
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write nodes
  fprintf(fp,"points = [");
  for (NodeIterator n(grid); !n.end(); ++n) {

	 p = n->coord();

	 if ( grid.type() == Grid::TRIANGLES ) {
		if ( n.last() )
		  fprintf(fp,"%.16f %.16f]';\n", p.x, p.y);
		else
		  fprintf(fp,"%.16f %.16f\n", p.x, p.y );
	 }
	 else {
		if ( n.last() )
		  fprintf(fp,"%.16f %.16f %.16f]';\n", p.x, p.y, p.z);
		else
		  fprintf(fp,"%.16f %.16f %.16f\n", p.x, p.y, p.z);
	 }
		
  }
  fprintf(fp,"\n");

  // Write cells
  fprintf(fp,"cells = [");
  for (CellIterator c(grid); !c.end(); ++c) {

	 for (NodeIterator n(c); !n.end(); ++n)
		fprintf(fp, "%d ", n->id() + 1);

	 if ( c.last() )
		fprintf(fp, "];\n");
	 else
		fprintf(fp, "\n");

  }
  fprintf(fp,"\n");

  // Write edges (to make the pdeplot routines happy)
  fprintf(fp,"edges = [1;2;0;0;0;0;0];\n\n");

  // Close file
  fclose(fp);
}
//-----------------------------------------------------------------------------
void MatlabFile::writeFunction(const Function& u,
										 const std::string& filename,
										 const std::string& name,
										 const std::string& label)
{
  // Write grid
  writeGrid(u.grid(), filename, name, label);

  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write vector
  fprintf(fp, "%s = [", name.c_str());
  for (NodeIterator n(u.grid()); !n.end(); ++n)
	 fprintf(fp, " %.16f", u(*n));
  fprintf(fp, " ];\n");
}
//-----------------------------------------------------------------------------
