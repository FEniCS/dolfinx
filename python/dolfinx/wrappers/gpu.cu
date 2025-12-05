// Copyright (C) 2025 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <dolfinx/la/gpu_cudss.h>
#include <dolfinx/la/gpu_cusparse.h>

#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <thrust/device_vector.h>

#include "dolfinx_wrappers/numpy_dtype.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
void declare_objects(nb::module_& m, const std::string& type)
{
  // Wrapper for a Vector allocated on-device
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

  // Wrapper for a MatrixCSR allocated on-device
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

  // Solver for Au=b
  using GPUSolver = dolfinx::la::cuda::cudssSolver<GPUMatrixCSR, GPUVector>;
  std::string pyclass_solver_name = std::string("GPU_Solver_") + type;
  nb::class_<GPUSolver>(m, pyclass_solver_name.c_str())
      .def(nb::init<GPUMatrixCSR&, GPUVector&, GPUVector&>(), nb::arg("A"),
           nb::arg("b"), nb::arg("u"))
      .def("analyze", &GPUSolver::analyze)
      .def("factorize", &GPUSolver::factorize)
      .def("solve", &GPUSolver::solve);

  // Operator for y=Ax
  using GPUSpmv = dolfinx::la::cuda::cusparseMatVec<GPUMatrixCSR, GPUVector>;
  std::string pyclass_spmv_name = std::string("GPU_SPMV_") + type;
  nb::class_<GPUSpmv>(m, pyclass_spmv_name.c_str())
      .def(nb::init<GPUMatrixCSR&, GPUVector&, GPUVector&>(), nb::arg("A"),
           nb::arg("y"), nb::arg("x"))
      .def("apply", &GPUSpmv::apply);
}


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
