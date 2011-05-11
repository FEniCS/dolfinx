// Copyright (C) 2007-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells 2007
// Modified by Johan Hake 2009
//
// First added:  2007-07-08
// Last changed: 2009-10-21

#ifndef __PERIODIC_BC_H
#define __PERIODIC_BC_H

#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include "BoundaryCondition.h"

namespace dolfin
{

  class Mesh;
  class SubDomain;
  class Form;
  class GenericMatrix;
  class GenericVector;

  /// This class specifies the interface for setting periodic boundary
  /// conditions for partial differential equations,
  ///
  ///    u(x) = u(F^{-1}(x)) on G,
  ///    u(x) = u(F(x))      on H,
  ///
  /// where F : H --> G is a map from a subdomain H to a subdomain G.
  ///
  /// A periodic boundary condition must be defined by the domain G
  /// and the map F pulling coordinates back from H to G. The domain
  /// and the map are both defined by a subclass of SubDomain which
  /// must overload both the inside() function, which specifies the
  /// points of G, and the map() function, which specifies the map
  /// from the points of H to the points of G.
  ///
  /// The implementation is based on matching degrees of freedom on G
  /// with degrees of freedom on H and only works when the mapping F
  /// is bijective between the sets of coordinates associated with the
  /// two domains. In other words, the nodes (degrees of freedom) must
  /// be aligned on G and H.
  ///
  /// The matching of degrees of freedom is done at the construction
  /// of the periodic boundary condition and is reused on subsequent
  /// applications to a linear system. The matching may be recomputed
  /// by calling the rebuild() function.

  class PeriodicBC : public BoundaryCondition
  {
  public:

    /// Create periodic boundary condition for sub domain
    PeriodicBC(const FunctionSpace& V,
               const SubDomain& sub_domain);

    /// Create periodic boundary condition for sub domain
    PeriodicBC(boost::shared_ptr<const FunctionSpace> V,
               boost::shared_ptr<const SubDomain> sub_domain);

    /// Destructor
    ~PeriodicBC();

    /// Apply boundary condition to a matrix
    void apply(GenericMatrix& A) const;

    /// Apply boundary condition to a vector
    void apply(GenericVector& b) const;

    /// Apply boundary condition to a linear system
    void apply(GenericMatrix& A, GenericVector& b) const;

    /// Apply boundary condition to a vector for a nonlinear problem
    void apply(GenericVector& b, const GenericVector& x) const;

    /// Apply boundary condition to a linear system for a nonlinear problem
    void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x) const;

    /// Rebuild mapping between dofs
    void rebuild();

  private:

    // Apply boundary conditions, common method
    void apply(GenericMatrix* A, GenericVector* b, const GenericVector* x) const;

    // Extract dof pairs for sub space and append to list
    void extract_dof_pairs(const FunctionSpace& function_space, std::vector<std::pair<uint, uint> >& dof_pairs);

    // The subdomain
    boost::shared_ptr<const SubDomain> sub_domain;

    // Number of dof pairs
    uint num_dof_pairs;

    // Array of master dofs (size num_dof_pairs)
    uint* master_dofs;

    // Array of slave dofs (size num_dof_pairs)
    uint* slave_dofs;

    // Right-hand side values, used for zeroing entries in right-hand side (size num_dof_pairs)
    double* rhs_values_master;
    double* rhs_values_slave;

  };

}

#endif
