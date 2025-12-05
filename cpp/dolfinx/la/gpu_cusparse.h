// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cassert>
#include <complex>
#include <cstdint>
#include <cusparse.h>

/// Interfaces to CUDA library functions
/// using dolfinx Vector and MatrixCSR which are
/// allocated on-device
namespace dolfinx::la::cuda
{

/// cusparse provides a sparse MatVec for CUDA
/// providing the operator  y = Ax
/// @tparam MatType Matrix type stored on-device
/// @tparam VecType Vector type stored on-device
template <typename MatType, typename VecType>
class cusparseMatVec
{

public:
  /// @brief Create an operator for y = Ax
  /// @param A_device CSR Matrix stored on device
  /// @param y_device Output Vector stored on device
  /// @param x_device Input Vector stored on device
  cusparseMatVec(MatType& A_device, VecType& y_device, VecType& x_device);

  /// Destructor
  ~cusparseMatVec();

  /// Apply the operator, computing y = Ax
  void apply();

private:
  // CUDA representation of scalar type
  cudaDataType_t data_type;

  // Data structures wrapping matrix
  cudaDataType cudaValueType;
  cusparseHandle_t handle;
  cusparseSpMatDescr_t matA;

  // Data structures wrapping vectors
  cusparseDnVecDescr_t vecX, vecY;

  // scratch
  void* dBuffer;

  // Coefficients in axpy
  MatType::value_type alpha, beta;
};
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
cusparseMatVec<MatType, VecType>::cusparseMatVec(MatType& A_device,
                                                 VecType& y_device,
                                                 VecType& x_device)
{
  using T = MatType::value_type;
  using U = VecType::value_type;
  static_assert(std::is_same_v<T, U>, "Incompatible data types");

  if constexpr (std::is_same_v<T, double>)
    data_type = CUDA_R_64F;
  else if constexpr (std::is_same_v<T, float>)
    data_type = CUDA_R_32F;
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    data_type = CUDA_C_32F;
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    data_type = CUDA_C_64F;
  else
    throw std::runtime_error("Value type not supported");

  cusparseCreate(&handle);

  int nnz = A_device.values().size();
  assert(A_device.values().size() == A_device.cols().size());
  int nrows = A_device.row_ptr().size() - 1;

  assert(nrows == y_device.array().size());
  int ncols = x_device.array().size();

  cusparseStatus_t status = cusparseCreateCsr(
      &matA, nrows, ncols, nnz, (void*)A_device.row_ptr().data().get(),
      (void*)A_device.cols().data().get(),
      (void*)A_device.values().data().get(), CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, data_type);
  assert(status == CUSPARSE_STATUS_SUCCESS);

  // Create dense vector X
  status = cusparseCreateDnVec(&vecX, x_device.array().size(),
                               x_device.array().data().get(), data_type);
  assert(status == CUSPARSE_STATUS_SUCCESS);
  // Create dense vector y
  status = cusparseCreateDnVec(&vecY, y_device.array().size(),
                               y_device.array().data().get(), data_type);
  assert(status == CUSPARSE_STATUS_SUCCESS);

  alpha = 1.0;
  beta = 0.0;

  // allocate an external buffer if needed
  std::size_t bufferSize;
  status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, vecX, &beta, vecY, data_type,
                                   CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  assert(status == CUSPARSE_STATUS_SUCCESS);

  cudaError_t err = cudaMalloc(&dBuffer, bufferSize);
  assert(err == cudaSuccess);
}
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
void cusparseMatVec<MatType, VecType>::apply()
{
  cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
               &beta, vecY, data_type, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
}
//-----------------------------------------------------------------------------
template <typename MatType, typename VecType>
cusparseMatVec<MatType, VecType>::~cusparseMatVec()
{
  cusparseStatus_t status = cusparseDestroySpMat(matA);
  assert(status == CUSPARSE_STATUS_SUCCESS);
  status = cusparseDestroy(handle);
  assert(status == CUSPARSE_STATUS_SUCCESS);
  status = cusparseDestroyDnVec(vecX);
  assert(status == CUSPARSE_STATUS_SUCCESS);
  status = cusparseDestroyDnVec(vecY);
  assert(status == CUSPARSE_STATUS_SUCCESS);
  cudaError_t err = cudaFree(dBuffer);
  assert(err == 0);
}
} // namespace dolfinx::la::cuda
