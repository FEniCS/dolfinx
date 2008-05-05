// Copyright (C) 2008 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2008-05-05

#include <dolfin/common/Array.h>
#include <dolfin/common/constants.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "ALE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void ALE::move(Mesh& mesh, Mesh& new_boundary,
               const MeshFunction<uint>& vertex_map,
               ALEMethod method)
{
  // Only implemented in 3D so far
  if (mesh.topology().dim() < 2 || mesh.topology().dim() > 3 )
    error("Mesh interpolation only implemented in 2D and 3D so far.");

  // Only Lagrange implemented
  if (method != lagrange)
    error("Only Lagrange mesh interpolation implemented so far.");

  // Extract old coordinates
  const uint dim = mesh.geometry().dim();
  const uint size = mesh.numVertices()*dim;
  real* new_x = new real[size];
  
  // Iterate over coordinates in mesh
  for (VertexIterator v(mesh); !v.end(); ++v)
    meanValue(new_x + v->index()*dim, dim, new_boundary, mesh, vertex_map, *v);

  // Update mesh coordinates
  for (VertexIterator v(mesh); !v.end(); ++v)
    memcpy(v->x(), new_x + v->index()*dim, dim*sizeof(real));
  
  delete [] new_x;
}
//-----------------------------------------------------------------------------
void ALE::meanValue(real* new_x, uint dim, Mesh& new_boundary,
                    Mesh& mesh, const MeshFunction<uint>& vertex_map,
                    Vertex& vertex)
{
  // Check if the point is on the boundary (no need to compute new coordinate)
  for (VertexIterator v(new_boundary); !v.end(); ++v)
  {
    if (vertex_map(*v) == vertex.index())
    {
      memcpy(new_x, v->x(), dim*sizeof(real));
      return;
    }
  }

  //const real* old_x = vertex.x();
  //cout << "Old x: " << old_x[0] << " " << old_x[1] <<" "<<old_x[2];
  
  const uint size = new_boundary.numVertices();
  real * d = new real[size];
  real ** u = new real * [size];

  // Compute distance d and direction vector u from x to all p
  for (VertexIterator v(new_boundary);  !v.end(); ++v)
  {
    // Old position of point x
    const real* x = vertex.x();
    
    // Old position of vertex v in boundary
    const real* p = mesh.geometry().x(vertex_map(*v));

    // Distance from x to each point at the boundary
    d[v->index()] = dist(p, x, dim);
          
    // Compute direction vector for p-x
    u[v->index()] = new real [dim];
    for (uint i=0; i<dim; i++)
      u[v->index()][i]=(p[i] - x[i]) / d[v->index()];
  }
  
  // Local arrays
  const uint num_vertices = new_boundary.topology().dim() + 1;
  real * w      = new real [num_vertices];
  real ** new_p = new real * [num_vertices];
  real * dCell  = new real [num_vertices];  
  real ** uCell = new real * [num_vertices];
  
  // Set new x to zero
  for (uint i = 0; i < dim; ++i)
    new_x[i] = 0.0;
  
  // Iterate over all cells in boundary
  real totalW = 0.0;
  for (CellIterator c(new_boundary); !c.end(); ++c)
  {
    // Get local data
    for (VertexIterator v(*c); !v.end(); ++v)
    {
      const uint ind = v.pos();
      new_p[ind] = v->x();
      uCell[ind] = u[v->index()];
      dCell[ind] = d[v->index()];
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
      for (uint i=0; i<num_vertices; i++)
	new_x[j] += w[i]*new_p[i][j];
      
    //psi[xnr]=1/totalW;
  }

  // Scale by totalW
  for (uint i = 0; i < dim; i++)
    new_x[i] /= totalW;

  //cout<<"  New x: " << new_x[0] << " " << new_x[1] << " " << new_x[2] << endl;

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
void ALE::computeWeights2D(real* w, real** u, real* d,
                           uint dim, uint num_vertices)
{  
  for (uint i=0; i < num_vertices; i++)
    w[i] = tan(asin(u[0][0]*u[1][1] - u[0][1]*u[1][0])/2) / d[i];
}
//-----------------------------------------------------------------------------
void ALE::computeWeights3D(real* w, real** u, real* d,
                           uint dim, uint num_vertices)
{
  Array<real> ell(num_vertices);
  Array<real> theta(num_vertices);
  real h = 0.0;
  
  for (uint i = 0; i < num_vertices; i++)
  {
    const uint ind1 = next(i, num_vertices);
    const uint ind2 = previous(i, num_vertices);

    ell[i] = dist(u[ind1], u[ind2], dim);
    		 
    theta[i] = 2*asin(ell[i] / 2.0);
    h += theta[i] / 2.0;
  }
    
  Array<real> c(num_vertices);
  Array<real> s(num_vertices);
  
  for (uint i = 0; i < num_vertices; i++)
  {
    const uint ind1 = next(i, num_vertices);
    const uint ind2 = previous(i, num_vertices);

    c[i] = (2*sin(h)*sin(h - theta[i])) / (sin(theta[ind1])*sin(theta[ind2])) - 1;
    const real sinus=1-c[i]*c[i];
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
real ALE::dist(const real* x, const real* y, uint dim)
{
  real s = 0.0;
  for (uint i = 0; i < dim; i++)
    s += (x[i] - y[i])*(x[i] - y[i]);
  return sqrt(s);
}
//-----------------------------------------------------------------------------
