// Copyright (C) 2011 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-03-29
// Last changed:

#include <boost/shared_ptr.hpp>
#include <dolfin/common/Array.h>
//#include <dolfin/common/Hierarchical.h>

#ifndef __GENERIC_FUNCTION_SPACE_H
#define __GENERIC_FUNCTION_SPACE_H

namespace dolfin
{

  class Mesh;
  class FiniteElement;
  class GenericDofMap;

  class GenericFunctionSpace //: public Hierarchical<GenericFunctionSpace>
  {
  protected:

    /// Create function space for given mesh, element and dofmap (shared data)
    GenericFunctionSpace(boost::shared_ptr<const Mesh> mesh,
                  boost::shared_ptr<const FiniteElement> element,
                  boost::shared_ptr<const GenericDofMap> dofmap);


    /// Create empty function space for later initialization. This
    /// constructor is intended for use by any sub-classes which need
    /// to construct objects before the initialisation of the base
    /// class. Data can be attached to the base class using
    /// FunctionSpace::attach(...).
    GenericFunctionSpace(boost::shared_ptr<const Mesh> mesh);

    /// Copy constructor
    GenericFunctionSpace(const GenericFunctionSpace& V);

    /// Assignment operator
    const GenericFunctionSpace& operator= (const GenericFunctionSpace& V);

    /// Attach data to an empty FunctionSpace
    void attach(boost::shared_ptr<const FiniteElement> element,
                boost::shared_ptr<const GenericDofMap> dofmap);

  public:

    /// Return mesh
    const Mesh& mesh() const
    { assert(_mesh); return *_mesh; }

    /// Return finite element
    const FiniteElement& element() const
    { assert(_element); return *_element; }

    /// Return dofmap
    const GenericDofMap& dofmap() const
    { assert(_dofmap); return *_dofmap; }

    /// Return dimension of function space
    unsigned int dim() const;

    /// Extract sub space for component
    virtual boost::shared_ptr<GenericFunctionSpace> operator[] (unsigned int i) const = 0;

    // FIXME: This should be part if SubSpace only
    /// Return component (relative to super space)
    virtual const Array<unsigned int>& component() const = 0;

  protected:

    // The mesh
    boost::shared_ptr<const Mesh> _mesh;

    // The finite element
    boost::shared_ptr<const FiniteElement> _element;

    // The dofmap
    boost::shared_ptr<const GenericDofMap> _dofmap;

  };
}

#endif
