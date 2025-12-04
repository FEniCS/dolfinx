// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cudss.h>

#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <thrust/device_vector.h>

#include "dolfinx_wrappers/numpy_dtype.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

template <typename MatType, typename VecType>
class cudssSolver
{

public:
  cudssSolver(MatType& A_device, VecType& b_device, VecType& u_device);

  // Destructor
  ~cudssSolver();

  void factorize();
  void solve();

private:
  cudssMatrix_t A_dss;
  cudssMatrix_t b_dss;
  cudssMatrix_t u_dss;

  cudssHandle_t handle;
  cudssConfig_t config;
  cudssData_t data;
};

template <typename MatType, typename VecType>
cudssSolver<MatType, VecType>::cudssSolver(MatType& A_device, VecType& b_device,
                                           VecType& u_device)
{
  using T = MatType::value_type;

  cudaDataType_t data_type = CUDA_R_32F;
  if constexpr (std::is_same_v<T, double>)
    data_type = CUDA_R_64F;

  cudssCreate(&handle);
  cudssConfigCreate(&config);
  cudssDataCreate(handle, &data);

  cudssMatrixCreateCsr(
      &A_dss, A_device.index_map(0)->size_local(),
      A_device.index_map(1)->size_local(), A_device.cols().size(),
      (void*)(A_device.row_ptr().data().get()), NULL,
      (void*)(A_device.cols().data().get()),
      (void*)(A_device.values().data().get()), CUDA_R_32I, data_type,
      CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);

  std::int64_t nrows = b_device.index_map()->size_local();
  cudssMatrixCreateDn(&b_dss, nrows, 1, nrows,
                      (void*)b_device.array().data().get(), data_type,
                      CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&u_dss, nrows, 1, nrows,
                      (void*)u_device.array().data().get(), data_type,
                      CUDSS_LAYOUT_COL_MAJOR);
}

template <typename MatType, typename VecType>
void cudssSolver<MatType, VecType>::factorize()
{
  cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A_dss, u_dss, b_dss);
  cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A_dss, u_dss,
               b_dss);
}

template <typename MatType, typename VecType>
void cudssSolver<MatType, VecType>::solve()
{
  cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A_dss, u_dss, b_dss);
}

template <typename MatType, typename VecType>
cudssSolver<MatType, VecType>::~cudssSolver()
{
  cudssConfigDestroy(config);
  cudssDataDestroy(handle, data);
  cudssMatrixDestroy(A_dss);
  cudssMatrixDestroy(u_dss);
  cudssMatrixDestroy(b_dss);
  cudssDestroy(handle);
}

template <typename T>
void declare_objects(nb::module_& m, const std::string& type)
{
  using GPUVector = dolfinx::la::Vector<T, thrust::device_vector<T>,
                                        thrust::device_vector<std::int32_t>>;
  std::string pyclass_vector_name = std::string("GPU_Vector_") + type;
  nb::class_<GPUVector>(m, pyclass_vector_name.c_str())
      .def(nb::init<std::shared_ptr<const dolfinx::common::IndexMap>, int>(),
           nb::arg("map"), nb::arg("bs"))
      .def(nb::init<const dolfinx::la::Vector<T>&>(), nb::arg("vec"))
      .def_prop_ro("dtype", [](const GPUVector&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_prop_ro("index_map", &GPUVector::index_map)
      .def_prop_ro("bs", &GPUVector::bs)
      .def_prop_ro(
          "array",
          [](GPUVector& self)
          {
            return nb::ndarray<T, nb::cupy, nb::device::cuda>(
                self.array().data().get(), {self.array().size()});
          },
          nb::rv_policy::reference_internal);

  using GPUMatrixCSR
      = dolfinx::la::MatrixCSR<T, thrust::device_vector<T>,
                               thrust::device_vector<std::int32_t>,
                               thrust::device_vector<std::int32_t>>;
  std::string pyclass_matrix_name = std::string("GPU_MatrixCSR_") + type;
  nb::class_<GPUMatrixCSR>(m, pyclass_matrix_name.c_str())
      .def(nb::init<const dolfinx::la::MatrixCSR<T>&>(), nb::arg("mat"))
      .def_prop_ro("dtype", [](const GPUMatrixCSR&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_prop_ro("index_map", &GPUMatrixCSR::index_map)
      .def_prop_ro("bs", &GPUMatrixCSR::block_size)
      .def_prop_ro(
          "data",
          [](GPUMatrixCSR& self)
          {
            return nb::ndarray<T, nb::cupy, nb::device::cuda>(
                self.values().data().get(), {self.values().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "indices",
          [](GPUMatrixCSR& self)
          {
            return nb::ndarray<const std::int32_t, nb::cupy, nb::device::cuda>(
                self.cols().data().get(), {self.cols().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "indptr",
          [](GPUMatrixCSR& self)
          {
            return nb::ndarray<const std::int32_t, nb::cupy, nb::device::cuda>(
                self.row_ptr().data().get(), {self.row_ptr().size()});
          },
          nb::rv_policy::reference_internal);

  using GPUSolver = cudssSolver<GPUMatrixCSR, GPUVector>;
  std::string pyclass_solver_name = std::string("GPU_Solver_") + type;
  nb::class_<GPUSolver>(m, pyclass_solver_name.c_str())
      .def(nb::init<GPUMatrixCSR&, GPUVector&, GPUVector&>(), nb::arg("A"),
           nb::arg("b"), nb::arg("u"))
      .def("factorize", &GPUSolver::factorize)
      .def("solve", &GPUSolver::solve);
}
} // namespace

namespace dolfinx_wrappers
{
void gpu(nb::module_& m)
{
  declare_objects<double>(m, "float64");
  declare_objects<float>(m, "float32");
  declare_objects<std::complex<float>>(m, "complex64");
  declare_objects<std::complex<double>>(m, "complex128");
}
} // namespace dolfinx_wrappers
