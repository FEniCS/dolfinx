// Copyright (C) 2008 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2008-05-29

#include <string.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/constants.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include "ALE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void ALE::move(Mesh& mesh, Mesh& new_boundary, ALEType method)
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
  real* new_x = new real[size];
  real ** ghat = new real * [new_boundary.numVertices()];;

  // If hermite, create dgdn
  if (method == hermite)
  {
    cout<<"hermite"<<endl;
    hermiteFunction(ghat, dim, new_boundary,
		    mesh, *vertex_map, *cell_map);
  }

  // Iterate over coordinates in mesh
  for (VertexIterator v(mesh); !v.end(); ++v)
    meanValue(new_x + v->index()*dim, dim, new_boundary, mesh, *vertex_map, *v, ghat, method);

  // Update mesh coordinates
  for (VertexIterator v(mesh); !v.end(); ++v)
    memcpy(v->x(), new_x + v->index()*dim, dim*sizeof(real));
  
  delete [] new_x;
  if (method==hermite)
    for (uint i=0; i<new_boundary.numVertices(); i++) 
      delete [] ghat[i];
  delete [] ghat;
}
//-----------------------------------------------------------------------------
void ALE::meanValue(real* new_x, uint dim, Mesh& new_boundary,
                    Mesh& mesh, const MeshFunction<uint>& vertex_map,
                    Vertex& vertex, real** ghat, ALEType method)
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
  real * w         = new real [num_vertices];
  real ** new_p    = new real * [num_vertices];
  real * dCell     = new real [num_vertices];  
  real ** uCell    = new real * [num_vertices];
  real * herm      = new real [num_vertices];
  real ** ghatCell = new real * [num_vertices];
  
  // Set new x to zero
  for (uint i = 0; i < dim; ++i) {
    new_x[i] = 0.0;
    if (method == hermite)
      herm[i] = 0.0;
  }  
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
      if (method == hermite)
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
    for (uint j=0; j<dim; j++) {
      for (uint i=0; i<num_vertices; i++) {
	new_x[j] += w[i]*new_p[i][j];
	if (method == hermite) {
	  herm[j] += w[i]*ghatCell[i][j];
	}
      }
    }
  }
  // Scale by totalW
  if (method == lagrange) 
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
void ALE::hermiteFunction(real ** ghat, uint dim, Mesh& new_boundary,
			  Mesh& mesh, const MeshFunction<uint>& vertex_map, 
			  const MeshFunction<uint>& cell_map)
{
  real ** dfdn = new real * [new_boundary.numVertices()];
  normals(dfdn, dim, new_boundary,
	  mesh, vertex_map, cell_map);

  real c;
  if (dim==2)
    c=2;
  else
    c=M_PI;

  //FAKTOREN c før dfdn, HVA VELGER VI DER?
  for (VertexIterator v(new_boundary); !v.end(); ++v) {
    ghat[v->index()]=new real [dim];
    integral(ghat[v->index()], dim, new_boundary,
	     mesh, vertex_map, *v);
    for (uint i=0; i<dim;i++) 
      ghat[v->index()][i]=c*dfdn[v->index()][i]-ghat[v->index()][i];
  }
  for (uint i=0; i<new_boundary.numVertices(); i++)
    delete [] dfdn[i];
  delete [] dfdn;
}
//-----------------------------------------------------------------------------
void ALE::normals(real** dfdn, uint dim, Mesh& new_boundary,
		  Mesh& mesh, const MeshFunction<uint>& vertex_map, 
		  const MeshFunction<uint>& cell_map){
  
  real** p=new real* [dim];
  real* n=new real[dim];
  
  for (VertexIterator v(new_boundary); !v.end(); ++v)
  {
    const uint ind = v.pos();
    dfdn[ind] = new real[dim];
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
    real len=length(dfdn[ind], dim);
    
    for (uint i=0; i<dim; i++)
      dfdn[ind][i]/=len; 
  }

  delete [] p;
  delete [] n;
}
//-----------------------------------------------------------------------------
void ALE::integral(real* new_x, uint dim, Mesh& new_boundary,
                    Mesh& mesh, const MeshFunction<uint>& vertex_map,
                    Vertex& vertex)
{
  const uint size = new_boundary.numVertices();
  real * d = new real[size];
  real ** u = new real * [size];

  // Compute distance d and direction vector u from x to all p
  for (VertexIterator v(new_boundary);  !v.end(); ++v)
  {
    uint ind=v->index();
    if(ind != vertex.index()) {
     
      // Old position of point x
      const real* x = mesh.geometry().x(vertex_map(vertex));
      
      // Old position of vertex v in boundary
      const real* p = mesh.geometry().x(vertex_map(*v));
     
      // Distance from x to each point at the boundary
      d[ind] = dist(p, x, dim);
      
      // Compute direction vector for p-x
      u[ind] = new real[dim];
      for (uint i=0; i<dim; i++) 
	u[ind][i]=(p[i] - x[i]) / d[ind];
    }
    else {
      d[ind]=0;
      u[ind] = new real[dim];
      for (uint i=0; i<dim; i++)
	u[ind][i]=0;
    }
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
  //real totalW = 0.0;
  for (CellIterator c(new_boundary); !c.end(); ++c)
  {
    uint inCell=0;

    // Get local data
    for (VertexIterator v(*c); !v.end(); ++v)
    {
      const uint ind = v.pos();
      if (v->index()==vertex.index())
	inCell=1;
      else {
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
real ALE::dist(const real* x, const real* y, uint dim)
{
  real s = 0.0;
  for (uint i = 0; i < dim; i++)
    s += (x[i] - y[i])*(x[i] - y[i]);
  return sqrt(s);
}
//-----------------------------------------------------------------------------
real ALE::length(const real* x, uint dim)
{
  real s = 0.0;
  for (uint i = 0; i < dim; i++)
    s += x[i]*x[i];
  return sqrt(s);
}
//-----------------------------------------------------------------------------
