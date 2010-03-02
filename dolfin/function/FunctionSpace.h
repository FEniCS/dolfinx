// Copyright (C) 2008-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008-2009.
// Modified by Kent-Andre Mardal, 2009.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2008-09-11
// Last changed: 2010-02-26

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <map>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/Variable.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/fem/FiniteElement.h>

namespace dolfin
{

  class Mesh;
  class Cell;
  class DofMap;
  class Function;
  class GenericFunction;
  class GenericVector;
  template <class T> class MeshFunction;

  /// This class represents a finite element function space defined by
  /// a mesh, a finite element, and a local-to-global mapping of the
  /// degrees of freedom (dofmap).

  class FunctionSpace : public Variable
  {
  public:

    /// Create function space for given mesh, element and dofmap (shared data)
    FunctionSpace(boost::shared_ptr<Mesh> mesh,
                  boost::shared_ptr<const FiniteElement> element,
                  boost::shared_ptr<const DofMap> dofmap);

    /// Create function space for given mesh, element and dofmap (shared data)
    FunctionSpace(boost::shared_ptr<const Mesh> mesh,
                  boost::shared_ptr<const FiniteElement> element,
                  boost::shared_ptr<const DofMap> dofmap);

    /// Copy constructor
    FunctionSpace(const FunctionSpace& V);

    /// Destructor
    virtual ~FunctionSpace();

    /// Assignment operator
    const FunctionSpace& operator= (const FunctionSpace& V);

    /// Return mesh
    const Mesh& mesh() const;

    /// Return finite element
    const FiniteElement& element() const;

    /// Return dofmap
    const DofMap& dofmap() const;

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

    /// Return function space with a new dof map
    boost::shared_ptr<FunctionSpace>
    collapse_sub_space(boost::shared_ptr<DofMap> dofmap) const;

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

    // FIXME: Restrictions are broken
    /*

    /// Attach restriction meshfunction
    void attach(MeshFunction<bool>& restriction);

    /// Create function space based on the restriction
    boost::shared_ptr<FunctionSpace> restriction(MeshFunction<bool>& restriction);

    // Evaluate restriction
    bool is_inside_restriction(uint c) const;
    */

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

  private:

    // Friends
    friend class Function;
    friend class AdaptiveObjects;

    // The mesh
    boost::shared_ptr<const Mesh> _mesh;

    // The finite element
    boost::shared_ptr<const FiniteElement> _element;

    // The dofmap
    boost::shared_ptr<const DofMap> _dofmap;

    // The component (for sub spaces)
    Array<uint> _component;

    // The restriction meshfunction
    boost::shared_ptr<const MeshFunction<bool> > _restriction;

    // Cache of sub spaces
    mutable std::map<std::string, boost::shared_ptr<FunctionSpace> > subspaces;

  };

}

#endif
