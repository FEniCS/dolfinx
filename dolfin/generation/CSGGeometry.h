// Copyright (C) 2012 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2012-04-11
// Last changed: 2012-04-13

#ifndef __CSG_GEOMETRY_H
#define __CSG_GEOMETRY_H

#include <boost/shared_ptr.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>

namespace dolfin
{

  /// Geometry described by Constructive Solid Geometry (CSG)

  class CSGGeometry : public Variable
  {
  public:

    /// Constructor
    CSGGeometry();

    /// Destructor
    virtual ~CSGGeometry();

    /// Return dimension of geometry
    virtual uint dim() const = 0;

    /// Informal string representation
    virtual std::string str(bool verbose) const = 0;

  };

}

#endif
