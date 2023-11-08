
#include <complex>
#include <nanobind/nanobind.h>

#pragma once

template <typename T>
nb::object numpy_dtype()
{
  auto np = nb::module_::import_("numpy");
  nb::object dtype;

  if constexpr (std::is_same_v<T, double>)
    dtype = np.attr("float64");
  else if constexpr (std::is_same_v<T, float>)
    dtype = np.attr("float32");
  else if constexpr (std::is_same_v<T, std::complex<double>>)
    dtype = np.attr("complex128");
  else if constexpr (std::is_same_v<T, std::complex<float>>)
    dtype = np.attr("complex64");
  return dtype;
}
