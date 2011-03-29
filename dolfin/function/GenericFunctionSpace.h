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
  public:

    //GenericFunctionSpace() : Hierarchical<GenericFunctionSpace>(*this) {}
    GenericFunctionSpace() {}

    /// Return mesh
    virtual const Mesh& mesh() const = 0;

    /// Return finite element
    virtual const FiniteElement& element() const = 0;

    /// Return dofmap
    virtual const GenericDofMap& dofmap() const = 0;

    /// Return dimension of function space
    virtual unsigned int dim() const = 0;

    /// Extract sub space for component
    virtual boost::shared_ptr<GenericFunctionSpace> operator[] (unsigned int i) const = 0;

    // FIXME: This should be part if SubSpace only
    /// Return component (relative to super space)
    virtual const Array<unsigned int>& component() const = 0;

  };
}

#endif
