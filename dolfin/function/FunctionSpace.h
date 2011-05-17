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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2008-2011.
// Modified by Kent-Andre Mardal, 2009.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2008-09-11
// Last changed: 2011-05-15

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/fem/FiniteElement.h>

namespace dolfin
{

  class Mesh;
  class Cell;
  class GenericDofMap;
  class Function;
  class GenericFunction;
  class GenericVector;
  template <class T> class MeshFunction;

  /// This class represents a finite element function space defined by
  /// a mesh, a finite element, and a local-to-global mapping of the
  /// degrees of freedom (dofmap).

  class FunctionSpace : public Variable, public Hierarchical<FunctionSpace>
  {
  public:

    /// Create function space for given mesh, element and dofmap (shared data)
    FunctionSpace(boost::shared_ptr<const Mesh> mesh,
                  boost::shared_ptr<const FiniteElement> element,
                  boost::shared_ptr<const GenericDofMap> dofmap);

  protected:

    /// Create empty function space for later initialization. This
    /// constructor is intended for use by any sub-classes which need
    /// to construct objects before the initialisation of the base
    /// class. Data can be attached to the base class using
    /// FunctionSpace::attach(...).
    FunctionSpace(boost::shared_ptr<const Mesh> mesh);

  public:

    /// Copy constructor
    FunctionSpace(const FunctionSpace& V);

    /// Destructor
    virtual ~FunctionSpace();

  protected:

    /// Attach data to an empty FunctionSpace
    void attach(boost::shared_ptr<const FiniteElement> element,
                boost::shared_ptr<const GenericDofMap> dofmap);

  public:

    /// Assignment operator
    const FunctionSpace& operator= (const FunctionSpace& V);

    /// Return mesh
    const Mesh& mesh() const;

    /// Return finite element
    const FiniteElement& element() const;

    /// Return dofmap
    const GenericDofMap& dofmap() const;

    /// Return dimension of function space
    uint dim() const;

    /// Interpolate function v into function space, returning the vector of
    /// expansion coefficients
    void interpolate(GenericVector& expansion_coefficients,
                     const GenericFunction& v) const;

    /// Extract sub space for component
    boost::shared_ptr<FunctionSpace> operator[] (uint i) const;

    /// Extract sub space for component
    boost::shared_ptr<FunctionSpace>
    extract_sub_space(const std::vector<uint>& component) const;

    /// Collapse a subspace and return a new function space
    boost::shared_ptr<FunctionSpace> collapse() const;

    /// Collapse a subspace and return a new function space and a map from new
    /// to old dofs
    boost::shared_ptr<FunctionSpace>
    collapse(boost::unordered_map<uint, uint>& collapsed_dofs) const;

    /// Check if function space has given cell
    bool has_cell(const Cell& cell) const
    { return &cell.mesh() == &(*_mesh); }

    /// Check if function space has given element
    bool has_element(const FiniteElement& element) const
    { return element.hash() == _element->hash(); }

    /// Return component (relative to super space)
    const std::vector<uint>& component() const;

    /// Return informal string representation (pretty-print)
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
    std::vector<uint> _component;

    // Cache of sub spaces
    mutable std::map<std::vector<uint>, boost::shared_ptr<FunctionSpace> > subspaces;

  };

}

#endif
