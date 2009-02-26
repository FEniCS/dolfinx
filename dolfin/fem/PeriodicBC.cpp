// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2007
//
// First added:  2007-07-08
// Last changed: 2008-12-03

#include <vector>
#include <map>

#include <dolfin/common/constants.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include "DofMap.h"
#include "UFCMesh.h"
#include "UFCCell.h"
#include "BoundaryCondition.h"
#include "PeriodicBC.h"

using namespace dolfin;

// Comparison operator for hashing coordinates. Note that two
// coordinates are considered equal if equal to within round-off.
struct lt_coordinate
{
  bool operator() (const std::vector<double>& x, const std::vector<double>& y) const
  {
    if (x.size() > (y.size() + DOLFIN_EPS))
      return false;
    
    for (unsigned int i = 0; i < x.size(); i++)
    {
      if (x[i] < (y[i] - DOLFIN_EPS))
        return true;
      else if (x[i] > (y[i] + DOLFIN_EPS))
        return false;
    }
    
    return false;
  }
};

//-----------------------------------------------------------------------------
PeriodicBC::PeriodicBC(const FunctionSpace& V,
                       const SubDomain& sub_domain)
  : BoundaryCondition(V), sub_domain(sub_domain)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PeriodicBC::~PeriodicBC()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericMatrix& A) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericVector& b) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericMatrix& A, GenericVector& b) const
{
  cout << "Applying periodic boundary conditions to linear system." << endl;

  // FIXME: Make this work for non-scalar subsystems, like vector-valued
  // FIXME: Lagrange where more than one per element is associated with
  // FIXME: each coordinate. Note that globally there may very well be
  // FIXME: more than one dof per coordinate (for conforming elements).

  // Get mesh and dofmap
  const Mesh& mesh = V->mesh();
  const DofMap& dofmap = V->dofmap();

  // Set geometric dimension (needed for SWIG interface)
  sub_domain._geometric_dimension = mesh.geometry().dim();

  // Table of mappings from coordinates to dofs
  std::map<std::vector<double>, std::pair<int, int>, lt_coordinate> coordinate_dofs;
  typedef std::map<std::vector<double>, std::pair<int, int>, lt_coordinate>::iterator iterator;
  std::vector<double> xx(mesh.geometry().dim());
  
  // std::vector used for mapping coordinates
  double* y = new double[mesh.geometry().dim()];
  for (uint i = 0; i < mesh.geometry().dim(); i++)
    y[i] = 0.0;

  // Create local data for application of boundary conditions
  BoundaryCondition::LocalData data(*V);

  // Make sure we have the facet - cell connectivity
  const uint D = mesh.topology().dim();
  mesh.init(D - 1, D);
  
  // Create UFC view of mesh
  UFCMesh ufc_mesh(mesh);
  
  // Iterate over the facets of the mesh
  Progress p("Applying periodic boundary conditions", mesh.size(D - 1));
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Get cell to which facet belongs (there may be two, but pick first)
    Cell cell(mesh, facet->entities(D)[0]);
    UFCCell ufc_cell(cell);

    // Get local index of facet with respect to the cell
    const uint local_facet = cell.index(*facet);

    // Tabulate dofs on cell
    dofmap.tabulate_dofs(data.cell_dofs, ufc_cell, cell.index());
    
    // Tabulate coordinates on cell
    dofmap.tabulate_coordinates(data.coordinates, ufc_cell);

    // Tabulate which dofs are on the facet
    dofmap.tabulate_facet_dofs(data.facet_dofs, local_facet);

    // Iterate over facet dofs
    for (uint i = 0; i < dofmap.num_facet_dofs(); i++)
    {
      // Get dof and coordinate of dof
      const uint local_dof = data.facet_dofs[i];
      const int global_dof = static_cast<int>(dofmap.offset() + data.cell_dofs[local_dof]);
      double* x = data.coordinates[local_dof];

      // Map coordinate from H to G
      for (uint j = 0; j < mesh.geometry().dim(); j++)
        y[j] = x[j];
      sub_domain.map(x, y);

      // Check if coordinate is inside the domain G or in H
      const bool on_boundary = facet->numEntities(D) == 1;
      if (sub_domain.inside(x, on_boundary))
      {
        // Coordinate x is in G
        //cout << "Inside: " << x[0] << " " << x[1] << endl;
        
        // Copy coordinate to std::vector
        for (uint j = 0; j < mesh.geometry().dim(); j++)
          xx[j] = x[j];

        // Check if coordinate exists from before
        iterator it = coordinate_dofs.find(xx);
        if (it != coordinate_dofs.end())
        {
          // Check that we don't have more than one dof per coordinate
          /*
          if (it->second.first != -1)
          {
            cout << "Coordinate: x =";
            for (uint j = 0; j < mesh.geometry().dim(); j++)
              cout << " " << xx[j];
            cout << endl;
            cout << "Degrees of freedom: " << it->second.first << " " << global_dof << endl;
            error("More than one dof associated with coordinate. Did you forget to specify the subsystem?");
          }
          */
          it->second.first = global_dof;
        }
        else
        {
          // Doesn't exist, so create new pair (with illegal second value)
          std::pair<int, int> dofs(global_dof, -1);
          coordinate_dofs[xx] = dofs;
        }
      }
      else if (sub_domain.inside(y, on_boundary))
      {
        // y = F(x) is in G, so coordinate x is in H
        //cout << "Mapped: " << x[0] << " " << x[1] << endl;
        
        // Copy coordinate to std::vector
        for (uint j = 0; j < mesh.geometry().dim(); j++)
          xx[j] = y[j];
        
        // Check if coordinate exists from before
        iterator it = coordinate_dofs.find(xx);
        if (it != coordinate_dofs.end())
        {
          // Check that we don't have more than one dof per coordinate
          /*
          if (it->second.second != -1)
          {
            cout << "Coordinate: x =";
            for (uint j = 0; j < mesh.geometry().dim(); j++)
              cout << " " << xx[j];
            cout << endl;
            cout << "Degrees of freedom: " << it->second.second << " " << global_dof << endl;
            error("More than one dof associated with coordinate. Did you forget to specify the subsystem?");
          }
          */
          it->second.second = global_dof;
        }
        else
        {
          // Doesn't exist, so create new pair (with illegal first value)
          std::pair<int, int> dofs(-1, global_dof);
          coordinate_dofs[xx] = dofs;
        }
      }
    }

    p++;
  }

  /*
  // Insert 1 at (dof0, dof1)
  uint* rows = new uint[coordinate_dofs.size()];
  uint i = 0;
  for (iterator it = coordinate_dofs.begin(); it != coordinate_dofs.end(); ++it)
  rows[i++] = static_cast<uint>(it->second.first);
  A.ident(coordinate_dofs.size(), rows);
  */

  // Insert -1 at (dof0, dof1) and 0 on right-hand side
  uint* rows = new uint[1];
  uint* cols = new uint[1];
  double* vals = new double[1];
  double* zero = new double[1];
  for (iterator it = coordinate_dofs.begin(); it != coordinate_dofs.end(); ++it)
  {
    // Check that we got both dofs
    const int dof0 = it->second.first;
    const int dof1 = it->second.second;
    
    //cout << dof0 << " " << dof1 <<endl;
    
    if (dof0 == -1 || dof1 == -1)
    {
      cout << "At coordinate: x =";
      for (uint j = 0; j < mesh.geometry().dim(); j++)
        cout << " " << it->first[j];
      cout << endl;
      error("Unable to find a pair of matching dofs for periodic boundary condition.");
    }

    //cout << "Setting periodic bc at x =";
    //for (uint j = 0; j < mesh.geometry().dim(); j++)
    //  cout << " " << it->first[j];
    //cout << ": " << dof0 << " " << dof1 << endl;
    
    // FIXME: Perhaps this can be done more efficiently?

    // Set x_i - x_j = 0
    rows[0] = static_cast<uint>(dof0);
    cols[0] = static_cast<uint>(dof1);
    vals[0] = -1;
    zero[0] = 0.0;

    std::vector<uint> columns;
    std::vector<double> values;
    
    // Add slave-dof-row to master-dof-row  
    A.getrow(dof0, columns, values);
    A.add(&values[0], 1, &cols[0], columns.size(), &columns[0]);

    // Add slave-dof-entry to master-dof-entry 
    values.resize(1);
    b.get(&values[0], 1, &rows[0]);
    b.add(&values[0], 1, &cols[0]);

    // Replace slave-dof equation by relation enforcing periodicity
    A.ident(1, rows);
    A.set(vals, 1, rows, 1, cols);
    b.set(zero, 1, rows);
  }
  
  // Cleanup
  delete [] rows;
  delete [] cols;
  delete [] vals;
  delete [] zero;
  delete [] y;

  // Apply changes
  A.apply();
  b.apply();
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericVector& b, const GenericVector& x) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
void PeriodicBC::apply(GenericMatrix& A,
                       GenericVector& b,
                       const GenericVector& x) const
{
  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
