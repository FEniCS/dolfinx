
#include<dolfinx/fem/Form.h>
#include<map>
#include<span>
#include<algorithm>

template <typename T>
std::map<std::pair<dolfinx::fem::IntegralType, int>,
         std::pair<std::span<const T>, int>>
py_to_cpp_coeffs(const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                py::array_t<T, py::array::c_style>>& coeffs)
{
  using Key_t = typename std::remove_reference_t<decltype(coeffs)>::key_type;
  std::map<Key_t, std::pair<std::span<const T>, int>> c;
  std::transform(coeffs.begin(), coeffs.end(), std::inserter(c, c.end()),
                 [](auto& e) -> typename decltype(c)::value_type
                 {
                   return {e.first,
                           {std::span(e.second.data(), e.second.size()),
                            e.second.shape(1)}};
                 });
  return c;
}
