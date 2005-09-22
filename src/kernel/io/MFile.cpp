// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-05-06
// Last changed: 2005-09-20

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Mesh.h>
#include <dolfin/Function.h>
#include <dolfin/Sample.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/MFile.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MFile::MFile(const std::string filename) : 
  GenericFile(filename)
{
  type = "Octave/MATLAB";
}
//-----------------------------------------------------------------------------
MFile::~MFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MFile::operator<<(Vector& x)
{
  // FIXME: BROKEN
  dolfin_error("This function needs to be updated to the new format.");
  /*
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write vector
  fprintf(fp, "%s = [", x.name().c_str());
  for (unsigned int i = 0; i < x.size(); i++)
    fprintf(fp, " %.15e", x(i));
  fprintf(fp, " ];\n");
  
  // Close file
  fclose(fp);

  // Increase the number of times we have saved the vector
  ++x;

  cout << "Saved vector " << x.name() << " (" << x.label()
       << ") to file " << filename << " in Octave/Matlab format." << endl;
  */
}
//-----------------------------------------------------------------------------
void MFile::operator<<(Mesh& mesh)
{
  Point p;
  
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Create a list if we save the mesh a second time
  if ( no_meshes == 1 )
  {
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
  if ( no_meshes == 0 )
    fprintf(fp,"points = [");
  else
    fprintf(fp,"points{%d} = [", no_meshes + 1);
  for (NodeIterator n(mesh); !n.end(); ++n) {
    
    p = n->coord();
    
    if ( mesh.type() == Mesh::triangles ) {
      if ( n.last() )
	fprintf(fp,"%.15f %.15f]';\n", p.x, p.y);
      else
	fprintf(fp,"%.15f %.15f\n", p.x, p.y );
    }
    else {
      if ( n.last() )
	fprintf(fp,"%.15f %.15f %.15f]';\n", p.x, p.y, p.z);
      else
	fprintf(fp,"%.15f %.15f %.15f\n", p.x, p.y, p.z);
    }
    
  }
  fprintf(fp,"\n");
  
  // Write cells
  if ( no_meshes == 0 )
    fprintf(fp,"cells = [");
  else
    fprintf(fp,"cells{%d} = [", no_meshes + 1);
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    for (NodeIterator n(c); !n.end(); ++n)
      fprintf(fp, "%d ", n->id() + 1);
    
    if ( c.last() )
      fprintf(fp, "]';\n");
    else
      fprintf(fp, "\n");
  }
  fprintf(fp,"\n");
  
  // Write edges (to make the pdeplot routines happy)
  if ( no_meshes == 0 )
    fprintf(fp,"edges = [1;2;0;0;0;0;0];\n\n");
  else
    fprintf(fp,"edges{%d} = [1;2;0;0;0;0;0];\n\n", no_meshes + 1);
  
  // Close file
  fclose(fp);

  // Increase the number of times we have saved the mesh
  // FIXME: Count number of meshes saved to this file, rather
  // than the number of times this specific mesh has been saved.
  ++mesh;
  
  // Increase the number of meshes save to this file
  no_meshes++;

  cout << "Saved mesh " << mesh.name() << " (" << mesh.label()
       << ") to file " << filename << " in Octave/Matlab format." << endl;
}
//-----------------------------------------------------------------------------
void MFile::operator<<(Function& u)
{
  uint num_components = 0;
  const FiniteElement& element = u.element();

  if ( element.rank() == 0 )
  {
    num_components = 1;
  }
  else if ( element.rank() == 1 )
  {
    num_components = element.tensordim(0);
  }
  else
    dolfin_error("Cannot handle tensor-valued functions.");

  // Write mesh the first time
  if ( u.number() == 0 )
    *this << u.mesh();
  
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Move old vector into list if we are saving a new value
  if ( u.number() == 1 )
  {
    fprintf(fp, "tmp = %s;\n", u.name().c_str());
    fprintf(fp, "clear %s\n", u.name().c_str());
    fprintf(fp, "%s{1} = tmp;\n", u.name().c_str());
    fprintf(fp, "clear tmp\n\n");
  }

  // Write vector
  if ( u.number() == 0 )
  {
    fprintf(fp, "%s = [", u.name().c_str());
    for (unsigned int i = 0; i < num_components; i++)
    { 
      for (NodeIterator n(u.mesh()); !n.end(); ++n)
	fprintf(fp, " %.15f", u(*n, i));
      fprintf(fp, ";");
    }
    fprintf(fp, " ]';\n\n");
  }
  else
  {
    fprintf(fp, "%s{%d} = [", u.name().c_str(), u.number() + 1);
    for (unsigned int i = 0; i < num_components; i++)
    { 
      for (NodeIterator n(u.mesh()); !n.end(); ++n)
	fprintf(fp, " %.15f", u(*n, i));
      fprintf(fp, ";");
    }
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
void MFile::operator<<(Sample& sample)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Initialize data structures first time
  if ( no_frames == 0 )
  {
    fprintf(fp, "t = [];\n");
    fprintf(fp, "%s = [];\n", sample.name().c_str());
    fprintf(fp, "k = [];\n");
    fprintf(fp, "r = [];\n");
    fprintf(fp, "\n");
  }
  
  // Save time
  fprintf(fp, "t = [t %.15e];\n", sample.t());

  // Save solution
  fprintf(fp, "tmp = [ ");
  for (unsigned int i = 0; i < sample.size(); i++)
    fprintf(fp, "%.15e ", sample.u(i));  
  fprintf(fp, "];\n");
  fprintf(fp, "%s = [%s tmp'];\n", 
	  sample.name().c_str(), sample.name().c_str());
  //fprintf(fp, "clear tmp;\n");

  // Save time steps
  fprintf(fp, "tmp = [ ");
  for (unsigned int i = 0; i < sample.size(); i++)
    fprintf(fp, "%.15e ", sample.k(i));
  fprintf(fp, "];\n");
  fprintf(fp, "k = [k tmp'];\n");
  //fprintf(fp, "clear tmp;\n");

  // Save residuals
  fprintf(fp, "tmp = [ ");
  for (unsigned int i = 0; i < sample.size(); i++)
    fprintf(fp, "%.15e ", sample.r(i));
  fprintf(fp, "];\n");
  fprintf(fp, "r = [r tmp'];\n");
  fprintf(fp, "clear tmp;\n");

  // Increase frame counter
  no_frames++;
  
  // Close file
  fclose(fp);
}
//-----------------------------------------------------------------------------
