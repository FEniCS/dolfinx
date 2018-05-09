// Copyright (C) 2008-2014 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#ifdef HAS_PETSC
#include <petscsys.h>
#endif

namespace dolfin
{

/// Index type for compatibility with linear algebra backend(s)
#ifdef HAS_PETSC
// typedef PetscInt la_index_t;
using la_index_t = PetscInt;
#else
// typedef std::int32_t la_index_t;
using la_index_t = std::int32_t;
#endif

// Typedefs for Eigen

// bool Arrays
using EigenArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
using EigenRowArrayXXb
    = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// double Arrays
using EigenArrayXd = Eigen::Array<double, Eigen::Dynamic, 1>;
using EigenRowArrayXd = Eigen::Array<double, 1, Eigen::Dynamic>;
using EigenRowArrayXXd
    = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// int32 Arrays
using EigenArrayXi32 = Eigen::Array<std::int32_t, Eigen::Dynamic, 1>;
using EigenRowArrayXi32 = Eigen::Array<std::int32_t, 1, Eigen::Dynamic>;
using EigenRowArrayXXi32 = Eigen::Array<std::int32_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>;

// int64 Arrays
using EigenArrayXi64 = Eigen::Array<std::int64_t, Eigen::Dynamic, 1>;
using EigenRowArrayXi64 = Eigen::Array<std::int64_t, 1, Eigen::Dynamic>;
using EigenRowArrayXXi64 = Eigen::Array<std::int64_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>;

// la_index_t Arrays
using EigenArrayXlaindex = Eigen::Array<dolfin::la_index_t, Eigen::Dynamic, 1>;

// double Matrices
using EigenRowMatrixXd
    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// double Vectors
using EigenVectorXd = Eigen::Matrix<double, Eigen::Dynamic, 1>;
} // namespace dolfin
