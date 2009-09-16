// Copyright (C) 2008-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008-2009.
// Modified by Kent-Andre Mardal, 2009.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2008-09-11
// Last changed: 2009-09-16

#ifndef __FUNCTION_SPACE_H
#define __FUNCTION_SPACE_H

#include <map>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>

namespace ufc
{
  class cell;
}

namespace dolfin
{

  class Mesh;
  class FiniteElement;
  class DofMap;
  class Function;
  class IntersectionDetector;
  class GenericVector;
  template <class T> class MeshFunction;

  /// This class represents a finite element function space defined by
  /// a mesh, a finite element, and a local-to-global mapping of the
  /// degrees of freedom (dofmap).

  class FunctionSpace
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

    /// Evaluate function v in function space at given point
    void eval(double* values,
              const double* x,
              const Function& v) const;

    /// Evaluate function v in function space at given point in given cell
    void eval(double* values,
              const double* x,
              const Function& v,
              const ufc::cell& ufc_cell,
              uint cell_index) const;

    /// Interpolate function v in function space (meshes = "matching" or "non-matching")
    void interpolate(GenericVector& coefficients,
                     const Function& v, std::string meshes = "matching") const;

    /// Interpolate function v in function space to vertices of mesh
    void interpolate_vertex_values(double* vertex_values,
                                   const Function& v) const;

    /// Extract sub space for component
    boost::shared_ptr<FunctionSpace> operator[] (uint i) const;

    /// Extract sub space for component
    boost::shared_ptr<FunctionSpace> extract_sub_space(const std::vector<uint>& component) const;

    /// Return function space with a new dof map
    boost::shared_ptr<FunctionSpace>
    collapse_sub_space(boost::shared_ptr<DofMap> dofmap) const;

    // Attach restriction meshfunction
    void attach(MeshFunction<bool>& restriction);

    // Create function space based on the restriction
    boost::shared_ptr<FunctionSpace> restriction(MeshFunction<bool>& restriction);

    // Evaluate restriction
    bool is_inside_restriction(uint c) const;

  private:

    // Scratch space, used for storing temporary local data
    class Scratch
    {
    public:

      // Constructor
      Scratch(const FiniteElement& element);

      // Constructor
      Scratch();

      // Destructor
      ~Scratch();

      // Initialize scratch space
      void init(const FiniteElement& element);

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
    boost::shared_ptr<const Mesh> _mesh;

    // The finite element
    boost::shared_ptr<const FiniteElement> _element;

    // The dofmap
    boost::shared_ptr<const DofMap> _dofmap;

    // The restriction meshfunction
    boost::shared_ptr<const MeshFunction<bool> > _restriction;

    // Cache of sub spaces
    mutable std::map<std::string, boost::shared_ptr<FunctionSpace> > subspaces;

    // Scratch space, used for storing temporary local data
    mutable Scratch scratch;

    // Intersection detector, used for evaluation at arbitrary points
    mutable IntersectionDetector* intersection_detector;

  };

}

#endif
