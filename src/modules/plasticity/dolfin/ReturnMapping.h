// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __RETURN_MAPPING_H
#define __RETURN_MAPPING_H

#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/ublas.h>
#include <dolfin/PlasticityModel.h>

namespace dolfin
{

  class ReturnMapping
  {
  public:

    /// Closest point projection return mapping
  static void ClosestPoint(PlasticityModel* plas, uBlasDenseMatrix& cons_t, uBlasDenseMatrix& D, 
        uBlasVector& t_sig, uBlasVector& eps_p, real& eps_eq);

  };
}

#endif
