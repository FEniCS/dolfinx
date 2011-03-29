// Copyright (C) 2008-2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008-2009.
// Modified by Kent-Andre Mardal, 2009.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2008-09-11
// Last changed: 2011-01-30

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <map>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/fem/FiniteElement.h>
#include "GenericFunctionSpace.h"

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

  class FunctionSpace : public Variable, public Hierarchical<FunctionSpace>, public GenericFunctionSpace
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
    boost::shared_ptr<GenericFunctionSpace> operator[] (uint i) const;

    /// Extract sub space for component
    boost::shared_ptr<FunctionSpace>
    extract_sub_space(const std::vector<uint>& component) const;

    /// Return function space with a new dof map
    boost::shared_ptr<FunctionSpace>
    collapse_sub_space(boost::shared_ptr<GenericDofMap> dofmap) const;

    /// Check if function space has given cell
    bool has_cell(const Cell& cell) const
    {
      return &cell.mesh() == &(*_mesh);
    }

    /// Check if function space has given element
    bool has_element(const FiniteElement& element) const
    {
      return element.hash() == _element->hash();
    }

    /// Return component (relative to super space)
    const Array<uint>& component() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Print dofmap (useful for debugging)
    void print_dofmap() const;

  private:

    // Friends
    friend class Function;

    // The mesh
    boost::shared_ptr<const Mesh> _mesh;

    // The finite element
    boost::shared_ptr<const FiniteElement> _element;

    // The dofmap
    boost::shared_ptr<const GenericDofMap> _dofmap;

    // The component (for sub spaces)
    Array<uint> _component;

    // Cache of sub spaces
    mutable std::map<std::string, boost::shared_ptr<FunctionSpace> > subspaces;

  };

}

#endif
