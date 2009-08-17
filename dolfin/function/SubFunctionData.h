// Copyright (C) 2007-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-27
// Last changed: 2009-08-17

#ifndef __SUB_FUNCTION_DATA_H
#define __SUB_FUNCTION_DATA_H

#include <boost/shared_ptr.hpp>
#include <dolfin/la/GenericVector.h>
#include "FunctionSpace.h"

namespace dolfin
{

  /// This class holds data for constructing a sub function (view) of a 
  /// function. Its purpose is to enable expressions like
  ///
  ///    Function w;
  ///    Function u = w[0];
  ///    Function p = w[1];
  ///
  /// without needing to create and destroy temporaries. 

  class SubFunctionData
  {
  
  friend class Function;

  private:

    SubFunctionData(boost::shared_ptr<const FunctionSpace> V, 
                boost::shared_ptr<GenericVector> x) 
              : _function_space(V), _vector(x) {}

  private:

    boost::shared_ptr<const FunctionSpace> _function_space;
    boost::shared_ptr<GenericVector> _vector;
  };

}

#endif
