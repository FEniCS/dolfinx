// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-09-11
// Last changed: 2008-10-09

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
  class NewFunction;

  /// This class represents a finite element function space defined by
  /// a mesh, a finite element, and a local-to-global mapping of the
  /// degrees of freedom (dofmap).

  class FunctionSpace
  {
  public:

    /// Create function space for given mesh, element and dofmap
    FunctionSpace(Mesh& mesh, const FiniteElement &element, const DofMap& dofmap);

    /// Create function space for given mesh, element and dofmap (may be shared)
    FunctionSpace(std::tr1::shared_ptr<Mesh> mesh,
                  std::tr1::shared_ptr<const FiniteElement> element,
                  std::tr1::shared_ptr<const DofMap> dofmap);

    /// Destructor
    ~FunctionSpace();

    /// Return mesh
    Mesh& mesh();

    /// Return mesh (const version)
    const Mesh& mesh() const;

    /// Return finite element
    const FiniteElement& element() const;

    /// Return dof map
    const DofMap& dofmap() const;

    /// Evaluate function v in function space at given point
    void eval(double* values,
              const double* x,
              const NewFunction& v) const;

    /// Interpolate function v in function space to local function space on cell
    void interpolate(double* coefficients,
                     const ufc::cell& cell,
                     const NewFunction& v);

    /// Interpolate function v in function space to global function space V
    void interpolate(GenericVector& coefficients,
                     const FunctionSpace& V,
                     const NewFunction& v);

    /// Interpolate function v in function space to vertices of mesh
    void interpolate(double* vertex_values,
                     const NewFunction& v);

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
      double* coefficients;
      
      // Local array for values
      double* values;

    };

    // The mesh
    std::tr1::shared_ptr<Mesh> _mesh;

    // The finite element
    std::tr1::shared_ptr<const FiniteElement> _element;

    // The dof map
    std::tr1::shared_ptr<const DofMap> _dofmap;

    // Scratch space, used for storing temporary local data
    mutable Scratch scratch;

    // Intersection detector, used for evaluation at arbitrary points
    mutable IntersectionDetector* intersection_detector;

  };

}

#endif
