// Copyright (C) 2008-2014 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#ifdef HAS_PETSC
#include <petscsys.h>
#endif

#include <Eigen/Dense>

// Some typedefs for common Eigen templates
using EigenRowMatrixXd
    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenVectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

namespace dolfin
{

/// Index type for compatibility with linear algebra backend(s)
#ifdef HAS_PETSC
typedef PetscInt la_index_t;
#else
typedef std::int32_t la_index_t;
#endif
}
