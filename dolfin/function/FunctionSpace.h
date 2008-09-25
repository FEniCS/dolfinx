// Copyright (C) 2008 Anders Logg (and others?).
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2008-09-25

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <tr1/memory>

namespace dolfin
{

  class Mesh;
  class FiniteElement;
  class DofMap;
  class IntersectionDetector;
  class GenericVector;

  /// This class represents a finite element function space
  /// defined by a mesh, a finite element, and local-to-global
  /// mapping (dof map).

  class FunctionSpace
  {
  public:

    /// Create function space for given mesh, finite element and dof map
    FunctionSpace(Mesh& mesh, const FiniteElement &element, const DofMap& dofmap);

    /// Create function space for given data (possibly shared)
    FunctionSpace(std::tr1::shared_ptr<Mesh> mesh,
                  std::tr1::shared_ptr<const FiniteElement> element,
                  std::tr1::shared_ptr<const DofMap> dofmap);

    /// Destructor
    ~FunctionSpace();

    /// Return mesh (non-const version)
    Mesh& mesh();

    /// Return mesh (const version)
    const Mesh& mesh() const;

    /// Return finite element
    const FiniteElement& element() const;

    /// Return dof map (const version)
    const DofMap& dofmap() const;

    /// Evaluate function with given vector of dofs at given point
    void eval(real* values, const real* p, const GenericVector& x) const;
    
  private:

    // Scratch space, used for storing temporary local data
    class Scratch
    {
    public:

      // Constructor
      Scratch(const FiniteElement& element);

      // Destructor
      ~Scratch();

      // Value size (number of entries in tensor value)
      uint size;
      
      // Local array for mapping of dofs
      uint* dofs;
      
      // Local array for expansion coefficients
      real* coefficients;
      
      // Local array for values
      real* values;

    };

    // The mesh
    std::tr1::shared_ptr<Mesh> _mesh;

    // The finite element
    std::tr1::shared_ptr<const FiniteElement> _element;

    // The dof map
    std::tr1::shared_ptr<const DofMap> _dofmap;

    // Scratch space, used for storing temporary local data
    mutable Scratch scratch;

    // Intersection detector
    mutable IntersectionDetector* intersection_detector;

  };

}

#endif
