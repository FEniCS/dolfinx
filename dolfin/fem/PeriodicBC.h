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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
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

#include <boost/serialization/map.hpp>
#include <dolfin/common/tuple_serialization.h>

namespace dolfin
{

  class Form;
  class GenericMatrix;
  class GenericVector;
  class Mesh;
  class SubDomain;

  /// This class specifies the interface for setting periodic boundary
  /// conditions for partial differential equations,
  ///
  /// .. math::
  ///
  ///     u(x) &= u(F^{-1}(x)) \hbox { on } G,
  ///
  ///     u(x) &= u(F(x))      \hbox{ on } H,
  ///
  /// where F : H --> G is a map from a subdomain H to a subdomain G.
  ///
  /// A periodic boundary condition must be defined by the domain G
  /// and the map F pulling coordinates back from H to G. The domain
  /// and the map are both defined by a subclass of _SubDomain_ which
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
  /// by calling the ``rebuild()`` function.

  class PeriodicBC : public BoundaryCondition
  {
  public:

    /// Create periodic boundary condition for subdomain
    ///
    /// *Arguments*
    ///     V (_FunctionSpace_)
    ///         The function space.
    ///     sub_domain (_SubDomain_)
    ///         The sub domain.
    PeriodicBC(const FunctionSpace& V,
               const SubDomain& sub_domain);

    /// Create periodic boundary condition for subdomain
    ///
    /// *Arguments*
    ///     V (_FunctionSpace_)
    ///         The function space.
    ///     sub_domain (_SubDomain_)
    ///         The subdomain.
    PeriodicBC(boost::shared_ptr<const FunctionSpace> V,
               boost::shared_ptr<const SubDomain> sub_domain);

    /// Destructor
    ~PeriodicBC();

    /// Apply boundary condition to a matrix
    ///
    /// *Arguments*
    ///     A (_GenericMatrix_)
    ///         The matrix to apply bc to.
    void apply(GenericMatrix& A) const;

    /// Apply boundary condition to a vector
    ///
    /// *Arguments*
    ///     b (_GenericVector_)
    ///         The vector to apply bc to.
    void apply(GenericVector& b) const;

    /// Apply boundary condition to a linear system
    ///
    /// *Arguments*
    ///     A (_GenericMatrix_)
    ///         The matrix.
    ///     b (_GenericVector_)
    ///         The vector.
    void apply(GenericMatrix& A, GenericVector& b) const;

    /// Apply boundary condition to a vector for a nonlinear problem
    ///
    /// *Arguments*
    ///     b (_GenericVector_)
    ///         The vector to apply bc to.
    ///     x (_GenericVector_)
    ///         Another vector (nonlinear problem).
    void apply(GenericVector& b, const GenericVector& x) const;

    /// Apply boundary condition to a linear system for a nonlinear
    /// problem
    ///
    /// *Arguments*
    ///     A (_GenericMatrix_)
    ///         The matrix to apply bc to.
    ///     b (_GenericVector_)
    ///         The vector to apply bc to.
    ///     x (_GenericVector_)
    ///         Another vector (nonlinear problem).
    void apply(GenericMatrix& A, GenericVector& b, const GenericVector& x) const;

    /// Return shared pointer to subdomain
    ///
    /// *Returns*
    ///     _SubDomain_
    ///         Shared pointer to subdomain.
    boost::shared_ptr<const SubDomain> sub_domain() const;

    /// Rebuild mapping between dofs
    void rebuild();

    // FIXME: This should find only pairs for which this process owns
    //        the slave dof
    /// Compute dof pairs (master dof, slave dof)
    std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > >
        compute_dof_pairs() const;

    // FIXME: This should find only pairs for which this process owns
    //        the slave dof
    /// Compute dof pairs (master dof, slave dof)
    void compute_dof_pairs(std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > >& dof_pairs) const;


  private:

    // Apply boundary conditions, common method
    void apply(GenericMatrix* A, GenericVector* b,
               const GenericVector* x) const;
    void parallel_apply(GenericMatrix* A, GenericVector* b,
                        const GenericVector* x) const;

    // FIXME: This should find only pairs for which this process owns
    //        the slave dof
    // Extract dof pairs for subspace and append to vector
    void extract_dof_pairs(const FunctionSpace& V,
      std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > >& dof_pairs) const;

    // The subdomain
    boost::shared_ptr<const SubDomain> _sub_domain;

    // Array of master dofs (size = num_dof_pairs)
    std::vector<uint> master_dofs;

    // Owners of master dofs in parallel (size = num_dof_pairs)
    std::vector<uint> master_owners;

    // Array of slave dofs (size = num_dof_pairs)
    std::vector<uint> slave_dofs;

    // Owners of slave dofs in parallel (size = num_dof_pairs)
    std::vector<uint> slave_owners;

  };

}

#endif
