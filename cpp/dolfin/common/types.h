// Copyright (C) 2008-2014 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <petscsys.h>

// Typedefs for ufc_scalar
#ifdef PETSC_USE_COMPLEX
#include <complex>
using ufc_scalar_t = std::complex<double>;
#else
using ufc_scalar_t = double;
#endif

namespace dolfin
{

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

// PetscInt Arrays
using EigenArrayXpetscint = Eigen::Array<PetscInt, Eigen::Dynamic, 1>;

// double Matrices
using EigenRowMatrixXd
    = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// double Vectors
using EigenVectorXd = Eigen::Matrix<double, Eigen::Dynamic, 1>;
} // namespace dolfin
