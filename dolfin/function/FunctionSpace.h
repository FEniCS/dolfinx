// Copyright (C) 2008 Anders Logg (and others?).
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2008-09-11

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <tr1/memory>

namespace dolfin
{

  class Mesh;
  class FiniteElement;
  class DofMap;

  /// This class represents a finite element function space
  /// defined by a mesh, a finite element and local-to-global
  /// mapping (dof map).

  class FunctionSpace
  {
  public:

    /// Create function space for given mesh, finite element and dof map
    FunctionSpace(Mesh& mesh, FiniteElement &element, DofMap& dofmap);

    /// Create function space for given data (possibly shared)
    FunctionSpace(std::tr1::shared_ptr<Mesh> mesh,
                  std::tr1::shared_ptr<FiniteElement> element,
                  std::tr1::shared_ptr<DofMap> dofmap);

    /// Destructor
    ~FunctionSpace();

    /// Return mesh
    Mesh& mesh();

    /// Return mesh (const version)
    const Mesh& mesh() const;

    /// Return finite element
    FiniteElement& element();

    /// Return finite element (const version)
    const FiniteElement& element() const;

    /// Return dof map
    DofMap& dofmap();

    /// Return dof map (const version)
    const DofMap& dofmap() const;

  private:

    // The mesh
    std::tr1::shared_ptr<Mesh> _mesh;

    // The finite element
    std::tr1::shared_ptr<FiniteElement> _element;

    // The dof map
    std::tr1::shared_ptr<DofMap> _dofmap;

  };

}

#endif
