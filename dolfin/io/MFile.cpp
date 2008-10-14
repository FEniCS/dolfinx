// Copyright (C) 2003-2008 Johan Hoffman and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2005
//
// First added:  2003-05-06
// Last changed: 2008-03-29

#include <dolfin/log/dolfin_log.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/function/Function.h>
#include <dolfin/ode/Sample.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "MFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MFile::MFile(const std::string filename) : 
  GenericFile(filename)
{
  type = "Octave/MATLAB";
  #ifdef HAS_GMP
    warning("MFile: Precision lost. Values will be saved with double precision");
  #endif

}
//-----------------------------------------------------------------------------
MFile::~MFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MFile::operator<<(GenericVector& x)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Write vector
  fprintf(fp, "x = [");
  double temp; 
  for (unsigned int i = 0; i < x.size(); i++)
  {
    // FIXME: This is a slow way to access PETSc vectors. Need a fast way 
    //        which is consistent for different vector types.
    x.get(&temp, 1, &i);
    fprintf(fp, " %.15g;", temp);
  }
  fprintf(fp, " ];\n");
  
  // Close file
  fclose(fp);

  message("Saved vector to file %s in Octave/MATLAB format.", filename.c_str());
}
//-----------------------------------------------------------------------------
void MFile::operator<<(Mesh& mesh)
{
  Point p;
  
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Create a list if we save the mesh a second time
  if ( counter == 1 )
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
  
  // Write vertices
  if ( counter == 0 )
    fprintf(fp,"points = [");
  else
    fprintf(fp,"points{%u} = [", counter + 1);
  for (VertexIterator v(mesh); !v.end();)
  {
    p = v->point();
    
    ++v;
    if ( mesh.type().cellType() == CellType::triangle )
    {
      if ( v.end() )
        fprintf(fp,"%.15f %.15f]';\n", p.x(), p.y());
      else
        fprintf(fp,"%.15f %.15f\n", p.x(), p.y() );
    }
    else {
      if ( v.end() )
        fprintf(fp,"%.15f %.15f %.15f]';\n", p.x(), p.y(), p.z());
      else
        fprintf(fp,"%.15f %.15f %.15f\n", p.x(), p.y(), p.z());
    }
  }
  fprintf(fp,"\n");
  
  // Write cells
  if ( counter == 0 )
    fprintf(fp,"cells = [");
  else
    fprintf(fp,"cells{%u} = [", counter + 1);
  for (CellIterator c(mesh); !c.end();)
  {
    for (VertexIterator v(*c); !v.end(); ++v)
      fprintf(fp, "%u ", (v->index()) + 1 );    
    ++c;
    if ( c.end() )
      fprintf(fp, "]';\n");
    else
      fprintf(fp, "\n");
  }
  fprintf(fp,"\n");

  // FIXME: Save all edges correctly
  // Write edges (to make the pdeplot routines happy)
  if ( counter == 0 )
    fprintf(fp,"edges = [1;2;0;0;0;0;0];\n\n");
  else
    fprintf(fp,"edges{%u} = [1;2;0;0;0;0;0];\n\n", counter + 1);
  
  // Close file
  fclose(fp);

//  // Increase the number of times we have saved the mesh
//  // FIXME: Count number of meshes saved to this file, rather
//  // than the number of times this specific mesh has been saved.
//  ++mesh;
  
  // Increase the number of meshes saved to this file
  counter++;

  message(1, "Saved mesh %s (%s) to file %s in Octave/MATLAB format.",
          mesh.name().c_str(), mesh.label().c_str(), filename.c_str());
}
//-----------------------------------------------------------------------------
void MFile::operator<<(Function& u)
{
  error("Function output in Matlab/Octave format not implemented for new Function.");
/*
  // Write mesh the first time
  if ( counter1 == 0 )
    *this << u.mesh();
  
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");
  
  // Move old vector into list if we are saving a new value
  if ( counter1 == 1 )
  {
    fprintf(fp, "tmp = %s;\n", u.name().c_str());
    fprintf(fp, "clear %s\n", u.name().c_str());
    fprintf(fp, "%s{1} = tmp;\n", u.name().c_str());
    fprintf(fp, "clear tmp\n\n");
  }

  // Write vector
  if ( counter1 == 0 )
  {
    fprintf(fp, "%s = [", u.name().c_str());
    for (unsigned int i = 0; i < u.vectordim(); i++)
    { 
      for (VertexIterator v(u.mesh()); !v.end(); ++v)
        fprintf(fp, " %.15f", u(*v, i));
        fprintf(fp, ";");
    }
    fprintf(fp, " ]';\n\n");
  }
  else
  {
    fprintf(fp, "%s{%u} = [", u.name().c_str(), counter1 + 1);
    for (unsigned int i = 0; i < u.vectordim(); i++)
    { 
      for (VertexIterator v(u.mesh()); !v.end(); ++v)
        fprintf(fp, " %.15f", u(*v, i));
        fprintf(fp, ";");
    }
    fprintf(fp, " ]';\n\n");
  }
  
  // Close file
  fclose(fp);
  
  // Increase the number of times we have saved the function
  counter1++;

  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in Octave/Matlab format." << endl;
*/
}
//-----------------------------------------------------------------------------
void MFile::operator<<(Sample& sample)
{
  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Initialize data structures first time
  if ( counter2 == 0 )
  {
    fprintf(fp, "t = [];\n");
    fprintf(fp, "%s = [];\n", sample.name().c_str());
    fprintf(fp, "k = [];\n");
    fprintf(fp, "r = [];\n");
    fprintf(fp, "\n");
  }
  
  // Save time
  fprintf(fp, "t = [t %.15e];\n", to_double(sample.t()));

  // Save solution
  fprintf(fp, "tmp = [ ");
  for (unsigned int i = 0; i < sample.size(); i++)
    fprintf(fp, "%.15e ", to_double(sample.u(i)));  
  fprintf(fp, "];\n");
  fprintf(fp, "%s = [%s tmp'];\n", 
	  sample.name().c_str(), sample.name().c_str());
  //fprintf(fp, "clear tmp;\n");

  // Save time steps
  fprintf(fp, "tmp = [ ");
  for (unsigned int i = 0; i < sample.size(); i++)
    fprintf(fp, "%.15e ", to_double(sample.k(i)));

  fprintf(fp, "];\n");
  fprintf(fp, "k = [k tmp'];\n");
  //fprintf(fp, "clear tmp;\n");

  // Save residuals
  fprintf(fp, "tmp = [ ");
  for (unsigned int i = 0; i < sample.size(); i++)
    fprintf(fp, "%.15e ", to_double(sample.r(i)));

  fprintf(fp, "];\n");
  fprintf(fp, "r = [r tmp'];\n");
  fprintf(fp, "clear tmp;\n");

  // Increase frame counter
  counter2++;
  
  // Close file
  fclose(fp);
}
//-----------------------------------------------------------------------------
