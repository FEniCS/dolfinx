// Copyright (C) 2008 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2008-09-11

#include <string.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/constants.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "TransfiniteInterpolation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void TransfiniteInterpolation::move(Mesh& mesh, Mesh& new_boundary, InterpolationType method)
{
  // Only implemented in 2D and 3D so far
  if (mesh.topology().dim() < 2 || mesh.topology().dim() > 3 )
    error("Mesh interpolation only implemented in 2D and 3D so far.");

  // Get vertex and cell maps
  const MeshFunction<uint>* vertex_map = new_boundary.data().meshFunction("vertex map");
  const MeshFunction<uint>* cell_map   = new_boundary.data().meshFunction("cell map");
  dolfin_assert(vertex_map);
  dolfin_assert(cell_map);

  // Extract old coordinates
  const uint dim = mesh.geometry().dim();
  const uint size = mesh.numVertices()*dim;
  double* new_x = new double[size];
  double ** ghat = new double * [new_boundary.numVertices()];;

  // If hermite, create dgdn
  if (method == interpolation_hermite)
  {
    hermiteFunction(ghat, dim, new_boundary,
		    mesh, *vertex_map, *cell_map);
  }

  // Iterate over coordinates in mesh
  for (VertexIterator v(mesh); !v.end(); ++v)
    meanValue(new_x + v->index()*dim, dim, new_boundary, mesh, *vertex_map, *v, ghat, method);

  // Update mesh coordinates
  MeshGeometry& geometry = mesh.geometry();
  for (uint i = 0; i < geometry.size(); i++)
    memcpy(geometry.x(i), new_x + i*dim, dim*sizeof(double));
  
  delete [] new_x;
  if (method == interpolation_hermite)
    for (uint i=0; i<new_boundary.numVertices(); i++) 
      delete [] ghat[i];
  delete [] ghat;
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::meanValue(double* new_x, uint dim, Mesh& new_boundary,
                                         Mesh& mesh, const MeshFunction<uint>& vertex_map,
                                         const Vertex& vertex, double** ghat, InterpolationType method)
{
  // Check if the point is on the boundary (no need to compute new coordinate)
  for (VertexIterator v(new_boundary); !v.end(); ++v)
  {
    if (vertex_map(*v) == vertex.index())
    {
      memcpy(new_x, v->x(), dim*sizeof(double));
      return;
    }
  }

  const uint size = new_boundary.numVertices();
  double * d = new double[size];
  double ** u = new double * [size];

  // Compute distance d and direction vector u from x to all p
  for (VertexIterator v(new_boundary);  !v.end(); ++v)
  {
    // Old position of point x
    const double* x = vertex.x();
    
    // Old position of vertex v in boundary
    const double* p = mesh.geometry().x(vertex_map(*v));

    // Distance from x to each point at the boundary
    d[v->index()] = dist(p, x, dim);
          
    // Compute direction vector for p-x
    u[v->index()] = new double [dim];
    for (uint i=0; i<dim; i++)
      u[v->index()][i]=(p[i] - x[i]) / d[v->index()];
  }
  
  // Local arrays
  const uint num_vertices = new_boundary.topology().dim() + 1;
  double* w         = new double  [num_vertices];
  double const** new_p = new double const * [num_vertices];
  double* dCell     = new double  [num_vertices];  
  double** uCell    = new double* [num_vertices];
  double* herm      = new double  [num_vertices];
  double** ghatCell = new double* [num_vertices];
  
  // Set new x to zero
  for (uint i = 0; i < dim; ++i) {
    new_x[i] = 0.0;
    if (method == interpolation_hermite)
      herm[i] = 0.0;
  }  
  // Iterate over all cells in boundary
  double totalW = 0.0;
  for (CellIterator c(new_boundary); !c.end(); ++c)
  {
    // Get local data
    for (VertexIterator v(*c); !v.end(); ++v)
    {
      const uint ind = v.pos();
      new_p[ind] = v->x();
      uCell[ind] = u[v->index()];
      dCell[ind] = d[v->index()];
      if (method == interpolation_hermite)
        ghatCell[ind]= ghat[v->index()];
    }
    
    // Compute weights w.
    if (mesh.topology().dim() == 2)
      computeWeights2D(w, uCell, dCell, dim, num_vertices);
    else
      computeWeights3D(w, uCell, dCell, dim, num_vertices);

    // Compute sum of weights
    for (uint i=0; i<num_vertices; i++)
      totalW += w[i];
    
    // Compute new position
    for (uint j=0; j<dim; j++) 
    {
      for (uint i=0; i<num_vertices; i++) 
      {
        new_x[j] += w[i]*new_p[i][j];
        if (method == interpolation_hermite) 
	        herm[j] += w[i]*ghatCell[i][j];
      }
    }
  }
  // Scale by totalW
  if (method == interpolation_lagrange) 
  {
    for (uint i = 0; i < dim; i++)
      new_x[i] /= totalW;
  }
  else
  {
    for (uint i = 0; i < dim; i++)
      new_x[i] = new_x[i]/totalW + herm[i]/(totalW*totalW);
  }

  //cout << "  New x: " << new_x[0] << " " << new_x[1] << " " << new_x[2] << endl;

  // Free memory for d
  delete [] d;
  
  // Free memory for u
  for (uint i = 0; i < size; ++i)
    delete [] u[i];
  delete [] u;
  
  // Free memory for local arrays
  delete [] w;
  delete [] new_p;
  delete [] uCell;
  delete [] dCell;
  delete [] herm;
  delete [] ghatCell;
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::computeWeights2D(double* w, double** u, double* d,
                           uint dim, uint num_vertices)
{  
  for (uint i=0; i < num_vertices; i++)
    w[i] = tan(asin(u[0][0]*u[1][1] - u[0][1]*u[1][0])/2) / d[i];
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::computeWeights3D(double* w, double** u, double* d,
                           uint dim, uint num_vertices)
{
  Array<double> ell(num_vertices);
  Array<double> theta(num_vertices);
  double h = 0.0;
  
  for (uint i = 0; i < num_vertices; i++)
  {
    const uint ind1 = next(i, num_vertices);
    const uint ind2 = previous(i, num_vertices);

    ell[i] = dist(u[ind1], u[ind2], dim);
    		 
    theta[i] = 2*asin(ell[i] / 2.0);
    h += theta[i] / 2.0;
  }
    
  Array<double> c(num_vertices);
  Array<double> s(num_vertices);
  
  for (uint i = 0; i < num_vertices; i++)
  {
    const uint ind1 = next(i, num_vertices);
    const uint ind2 = previous(i, num_vertices);

    c[i] = (2*sin(h)*sin(h - theta[i])) / (sin(theta[ind1])*sin(theta[ind2])) - 1;
    const double sinus=1-c[i]*c[i];
    if (sinus < 0 || sqrt(sinus) < DOLFIN_EPS)
    {
      for (uint i = 0; i < num_vertices; i++) 
        w[i]=0;
      return;
    }
    s[i] = sgn(det(u[0], u[1], u[2]))*sqrt(sinus);
  }
  
  for (uint i = 0; i < num_vertices; i++)
  {
    const uint ind1 = next(i, num_vertices);
    const uint ind2 = previous(i, num_vertices);
    
    w[i]=(theta[i]-c[ind1]*theta[ind2]-c[ind2]*theta[ind1])/(d[i]*sin(theta[ind1])*s[ind2]);
  } 
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::hermiteFunction(double** ghat, uint dim, Mesh& new_boundary,
			  Mesh& mesh, const MeshFunction<uint>& vertex_map, 
			  const MeshFunction<uint>& cell_map)
{
  double** dfdn = new double * [new_boundary.numVertices()];
  normals(dfdn, dim, new_boundary, mesh, vertex_map, cell_map);

  double c = 0.0;
  if (dim == 2)
    c = 2.0;
  else
    c = DOLFIN_PI;

  // FIXME *All* comments should be in English
  //FAKTOREN c før dfdn, HVA VELGER VI DER?
  for (VertexIterator v(new_boundary); !v.end(); ++v) 
  {
    ghat[v->index()] = new double [dim];
    integral(ghat[v->index()], dim, new_boundary, mesh, vertex_map, *v);
    for (uint i=0; i<dim;i++) 
      ghat[v->index()][i]=c*dfdn[v->index()][i]-ghat[v->index()][i];
  }
  for (uint i=0; i<new_boundary.numVertices(); i++)
    delete [] dfdn[i];
  delete [] dfdn;
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::normals(double** dfdn, uint dim, Mesh& new_boundary,
		  Mesh& mesh, const MeshFunction<uint>& vertex_map, 
		  const MeshFunction<uint>& cell_map){
  
  double** p = new double* [dim];
  double* n  = new double[dim];
  
  for (VertexIterator v(new_boundary); !v.end(); ++v)
  {
    const uint ind = v.pos();
    dfdn[ind] = new double[dim];
    for (uint i = 0; i < dim; i++) 
      dfdn[ind][i] = 0; 
    
    for (CellIterator c(new_boundary); !c.end(); ++c)
    {
      for (VertexIterator w(*c); !w.end(); ++w)
      {
        if(v->index() == w->index()) 
        {
          Facet mesh_facet(mesh, cell_map(*c));
          Cell mesh_cell(mesh, mesh_facet.entities(mesh.topology().dim())[0]);
          const uint facet_index = mesh_cell.index(mesh_facet);
          Point n=mesh_cell.normal(facet_index);
          
          for (uint j=0; j<dim; j++)
            dfdn[ind][j]-=n[j];
          break;
        }
      }
    }
    double len=length(dfdn[ind], dim);
    
    for (uint i=0; i<dim; i++)
      dfdn[ind][i]/=len; 
  }

  delete [] p;
  delete [] n;
}
//-----------------------------------------------------------------------------
void TransfiniteInterpolation::integral(double* new_x, uint dim, Mesh& new_boundary,
                    Mesh& mesh, const MeshFunction<uint>& vertex_map,
                    const Vertex& vertex)
{
  const uint size = new_boundary.numVertices();
  double * d = new double[size];
  double ** u = new double * [size];

  // Compute distance d and direction vector u from x to all p
  for (VertexIterator v(new_boundary);  !v.end(); ++v)
  {
    uint ind=v->index();
    if(ind != vertex.index()) 
    {
     
      // Old position of point x
      const double* x = mesh.geometry().x(vertex_map(vertex));
      
      // Old position of vertex v in boundary
      const double* p = mesh.geometry().x(vertex_map(*v));
     
      // Distance from x to each point at the boundary
      d[ind] = dist(p, x, dim);
      
      // Compute direction vector for p-x
      u[ind] = new double[dim];
      for (uint i=0; i<dim; i++) 
        u[ind][i]=(p[i] - x[i]) / d[ind];
    }
    else 
    {
      d[ind]=0;
      u[ind] = new double[dim];
      for (uint i=0; i<dim; i++)
        u[ind][i]=0;
    }
  }
  
  // Local arrays
  const uint num_vertices = new_boundary.topology().dim() + 1;
  double* w      = new double [num_vertices];
  double const** new_p = new double const* [num_vertices];
  double* dCell  = new double [num_vertices];  
  double** uCell = new double * [num_vertices];
  
  // Set new x to zero
  for (uint i = 0; i < dim; ++i)
    new_x[i] = 0.0;
    
  // Iterate over all cells in boundary
  //double totalW = 0.0;
  for (CellIterator c(new_boundary); !c.end(); ++c)
  {
    uint inCell=0;

    // Get local data
    for (VertexIterator v(*c); !v.end(); ++v)
    {
      const uint ind = v.pos();
      if (v->index()==vertex.index())
        inCell=1;
      else 
      {
        new_p[ind] = v->x();
        uCell[ind] = u[v->index()];
        dCell[ind] = d[v->index()];
      }
    }
    
    if (!inCell)
    {
      // Compute weights w.
      if (mesh.topology().dim() == 2)
        computeWeights2D(w, uCell, dCell, dim, num_vertices);
      else
        computeWeights3D(w, uCell, dCell, dim, num_vertices);
       
      // Compute new position
      for (uint j=0; j<dim; j++)
        for (uint i=0; i<num_vertices; i++)
          new_x[j] += w[i]*(new_p[i][j]-vertex.x()[j]);     
    }     
  }

  // Free memory for d
  delete [] d;
  
  // Free memory for u
  for (uint i = 0; i < size; ++i)
    delete [] u[i];
  delete [] u;
  
  // Free memory for local arrays
  delete [] w;
  delete [] new_p;
  delete [] uCell;
  delete [] dCell;
}
//-----------------------------------------------------------------------------
double TransfiniteInterpolation::dist(const double* x, const double* y, uint dim)
{
  double s = 0.0;
  for (uint i = 0; i < dim; i++)
    s += (x[i] - y[i])*(x[i] - y[i]);
  return sqrt(s);
}
//-----------------------------------------------------------------------------
double TransfiniteInterpolation::length(const double* x, uint dim)
{
  double s = 0.0;
  for (uint i = 0; i < dim; i++)
    s += x[i]*x[i];
  return sqrt(s);
}
//-----------------------------------------------------------------------------
