// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Grid.h>
#include <dolfin/Function.h>
#include <dolfin/MFile.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void MFile::operator>>(Vector& x)
{
  dolfin_warning("Cannot read vectors from Octave/Matlab files.");
}
//-----------------------------------------------------------------------------
void MFile::operator>>(Matrix& A)
{
  dolfin_warning("Cannot read matrices from Octave/Matlab files.");
}
//-----------------------------------------------------------------------------
void MFile::operator>>(Grid& grid)
{
  dolfin_warning("Cannot read grids from Octave/Matlab files.");
}
//-----------------------------------------------------------------------------
void MFile::operator>>(Function& u)
{
  dolfin_warning("Cannot read functions from Octave/Matlab files.");
}
//-----------------------------------------------------------------------------
void MFile::operator<<(Vector& x)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write vector
  fprintf(fp, "%s = [", x.name().c_str());
  for (int i = 0; i < x.size(); i++)
    fprintf(fp, " %.16e", x(i));
  fprintf(fp, " ];\n");
  
  // Close file
  fclose(fp);

  // Increase the number of times we have saved the vector
  ++x;

  cout << "Saved vector " << x.name() << " (" << x.label()
       << ") to file " << filename << " in Octave/Matlab format." << endl;
}
//-----------------------------------------------------------------------------
void MFile::operator<<(Grid& grid)
{
  Point p;
  
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Create a list if we save the grid a second time
  if ( no_grids == 1 ) {

    fprintf(fp, "tmp = points;\n");
    fprintf(fp, "clear points\n");
    fprintf(fp, "points{1} = tmp;\n");
    fprintf(fp, "clear tmp\n\n");

    fprintf(fp, "tmp = cells;\n");
    fprintf(fp, "clear cells\n");
    fprintf(fp, "cells{1} = tmp;\n");
    fprintf(fp, "clear tmp\n\n");

    fprintf(fp, "tmp = edges;\n");
    fprintf(fp, "clear edges\n");
    fprintf(fp, "edges{1} = tmp;\n");
    fprintf(fp, "clear tmp\n\n");

  }
  
  // Write nodes
  if ( no_grids == 0 )
    fprintf(fp,"points = [");
  else
    fprintf(fp,"points{%d} = [", no_grids + 1);
  for (NodeIterator n(grid); !n.end(); ++n) {
    
    p = n->coord();
    
    if ( grid.type() == Grid::triangles ) {
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
  if ( no_grids == 0 )
    fprintf(fp,"cells = [");
  else
    fprintf(fp,"cells{%d} = [", no_grids + 1);
  for (CellIterator c(grid); !c.end(); ++c) {
    
    for (NodeIterator n(c); !n.end(); ++n)
      fprintf(fp, "%d ", n->id() + 1);
    
    if ( c.last() )
      fprintf(fp, "]';\n");
    else
      fprintf(fp, "\n");
    
  }
  fprintf(fp,"\n");
  
  // Write edges (to make the pdeplot routines happy)
  if ( no_grids == 0 )
    fprintf(fp,"edges = [1;2;0;0;0;0;0];\n\n");
  else
    fprintf(fp,"edges{%d} = [1;2;0;0;0;0;0];\n\n", no_grids + 1);
  
  // Close file
  fclose(fp);

  // Increase the number of times we have saved the grid
  // FIXME: Count number of grids saved to this file, rather
  // than the number of times this specific grid has been saved.
  ++grid;
  
  // Increase the number of grids save to this file
  no_grids++;

  cout << "Saved grid " << grid.name() << " (" << grid.label()
       << ") to file " << filename << " in Octave/Matlab format." << endl;
}
//-----------------------------------------------------------------------------
void MFile::operator<<(Function& u)
{
  // Write grid the first time
  if ( u.number() == 0 )
    *this << u.grid();
  
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Move old vector into list if we are saving a new value
  if ( u.number() == 1 ) {
    fprintf(fp, "tmp = %s;\n", u.name().c_str());
    fprintf(fp, "clear %s\n", u.name().c_str());
    fprintf(fp, "%s{1} = tmp;\n", u.name().c_str());
    fprintf(fp, "clear tmp\n\n");
  }

  // Write vector
  if ( u.number() == 0 ) {
    fprintf(fp, "%s = [", u.name().c_str());
    for (NodeIterator n(u.grid()); !n.end(); ++n)
      fprintf(fp, " %.16f", u(*n));
    fprintf(fp, " ]';\n\n");
  }
  else {
    fprintf(fp, "%s{%d} = [", u.name().c_str(), u.number() + 1);
    for (NodeIterator n(u.grid()); !n.end(); ++n)
      fprintf(fp, " %.16f", u(*n));
    fprintf(fp, " ]';\n\n");
  }
  
  // Close file
  fclose(fp);
  
  // Increase the number of times we have saved the function
  ++u;

  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in Octave/Matlab format." << endl;
}
//-----------------------------------------------------------------------------
