// Copyright (C) 2008-2011 Anders Logg
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
// Modified by Garth N. Wells 2008-2011
// Modified by Kent-Andre Mardal 2009
// Modified by Ola Skavhaug 2009
//
// First added:  2008-09-11
// Last changed: 2012-11-02

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <cstddef>
#include <map>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <dolfin/common/Array.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/fem/FiniteElement.h>

namespace dolfin
{

  class Mesh;
  class Cell;
  class GenericDofMap;
  class Function;
  class GenericFunction;
  class GenericVector;
  template <typename T> class MeshFunction;

  /// This class represents a finite element function space defined by
  /// a mesh, a finite element, and a local-to-global mapping of the
  /// degrees of freedom (dofmap).

  class FunctionSpace : public Variable, public Hierarchical<FunctionSpace>
  {
  public:

    /// Create function space for given mesh, element and dofmap
    /// (shared data)
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    ///     element (_FiniteElement_)
    ///         The element.
    ///     dofmap (_GenericDofMap_)
    ///         The dofmap.
    FunctionSpace(boost::shared_ptr<const Mesh> mesh,
                  boost::shared_ptr<const FiniteElement> element,
                  boost::shared_ptr<const GenericDofMap> dofmap);

  protected:

    /// Create empty function space for later initialization. This
    /// constructor is intended for use by any sub-classes which need
    /// to construct objects before the initialisation of the base
    /// class. Data can be attached to the base class using
    /// FunctionSpace::attach(...).
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         The mesh.
    explicit FunctionSpace(boost::shared_ptr<const Mesh> mesh);

  public:

    /// Copy constructor
    ///
    /// *Arguments*
    ///     V (_FunctionSpace_)
    ///         The object to be copied.
    FunctionSpace(const FunctionSpace& V);

    /// Destructor
    virtual ~FunctionSpace();

  protected:

    /// Attach data to an empty function space
    ///
    /// *Arguments*
    ///     element (_FiniteElement_)
    ///         The element.
    ///     dofmap (_GenericDofMap_)
    ///         The dofmap.
    void attach(boost::shared_ptr<const FiniteElement> element,
                boost::shared_ptr<const GenericDofMap> dofmap);

  public:

    /// Assignment operator
    ///
    /// *Arguments*
    ///     V (_FunctionSpace_)
    ///         Another function space.
    const FunctionSpace& operator= (const FunctionSpace& V);

    /// Equality operator
    ///
    /// *Arguments*
    ///     V (_FunctionSpace_)
    ///         Another function space.
    bool operator== (const FunctionSpace& V) const;

    /// Unequality operator
    ///
    /// *Arguments*
    ///     V (_FunctionSpace_)
    ///         Another function space.
    bool operator!= (const FunctionSpace& V) const;

    /// Return mesh
    ///
    /// *Returns*
    ///     _Mesh_
    ///         The mesh.
    boost::shared_ptr<const Mesh> mesh() const;

    /// Return finite element
    ///
    /// *Returns*
    ///     _FiniteElement_
    ///         The finite element.
    boost::shared_ptr<const FiniteElement> element() const;

    /// Return dofmap
    ///
    /// *Returns*
    ///     _GenericDofMap_
    ///         The dofmap.
    boost::shared_ptr<const GenericDofMap> dofmap() const;

    /// Return dimension of function space
    ///
    /// *Returns*
    ///     std::size_t
    ///         The dimension of the function space.
    std::size_t dim() const;

    /// Interpolate function v into function space, returning the
    /// vector of expansion coefficients
    ///
    /// *Arguments*
    ///     expansion_coefficients (_GenericVector_)
    ///         The expansion coefficients.
    ///     v (_GenericFunction_)
    ///         The function to be interpolated.
    void interpolate(GenericVector& expansion_coefficients,
                     const GenericFunction& v) const;

    /// Extract subspace for component
    ///
    /// *Arguments*
    ///     i (std::size_t)
    ///         Index of the subspace.
    /// *Returns*
    ///     _FunctionSpace_
    ///         The subspace.
    boost::shared_ptr<FunctionSpace> operator[] (std::size_t i) const;

    /// Extract subspace for component
    ///
    /// *Arguments*
    ///     component (std::vector<std::size_t>)
    ///         The component.
    ///
    /// *Returns*
    ///     _FunctionSpace_
    ///         The subspace.
    boost::shared_ptr<FunctionSpace>
    extract_sub_space(const std::vector<std::size_t>& component) const;

    /// Collapse a subspace and return a new function space
    ///
    /// *Returns*
    ///     _FunctionSpace_
    ///         The new function space.
    boost::shared_ptr<FunctionSpace> collapse() const;

    /// Collapse a subspace and return a new function space and a map
    /// from new to old dofs
    ///
    /// *Arguments*
    ///     collapsed_dofs (boost::unordered_map<std::size_t, std::size_t>)
    ///         The map from new to old dofs.
    ///
    /// *Returns*
    ///     _FunctionSpace_
    ///       The new function space.
    boost::shared_ptr<FunctionSpace>
    collapse(boost::unordered_map<std::size_t, std::size_t>& collapsed_dofs) const;

    /// Check if function space has given cell
    ///
    /// *Arguments*
    ///     cell (_Cell_)
    ///         The cell.
    ///
    /// *Returns*
    ///     bool
    ///         True if the function space has the given cell.
    bool has_cell(const Cell& cell) const
    { return &cell.mesh() == &(*_mesh); }

    /// Check if function space has given element
    ///
    /// *Arguments*
    ///     element (_FiniteElement_)
    ///         The finite element.
    ///
    /// *Returns*
    ///     bool
    ///         True if the function space has the given element.
    bool has_element(const FiniteElement& element) const
    { return element.hash() == _element->hash(); }

    /// Return component
    ///
    /// *Returns*
    ///     std::vector<std::size_t>
    ///         The component (relative to superspace).
    std::vector<std::size_t> component() const;

    /// Return informal string representation (pretty-print)
    ///
    /// *Arguments*
    ///     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// *Returns*
    ///     std::string
    ///         An informal representation of the function space.
    std::string str(bool verbose) const;

    /// Print dofmap (useful for debugging)
    void print_dofmap() const;

  private:

    // The mesh
    boost::shared_ptr<const Mesh> _mesh;

    // The finite element
    boost::shared_ptr<const FiniteElement> _element;

    // The dofmap
    boost::shared_ptr<const GenericDofMap> _dofmap;

    // The component (for sub spaces)
    std::vector<std::size_t> _component;

    // Cache of sub spaces
    mutable std::map<std::vector<std::size_t>, boost::shared_ptr<FunctionSpace> > subspaces;

  };

}

#endif
