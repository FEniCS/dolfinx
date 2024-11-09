// ```text
// Copyright (C) 2022-2023 Garth N. Wells
// This file is part of DOLFINx (https://www.fenicsproject.org)
// SPDX-License-Identifier:    LGPL-3.0-or-later
// ```

// # Interpolation and IO

#include <basix/finite-element.h>
#include <cmath>
#include <concepts>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <filesystem>
#include <mpi.h>
#include <numbers>

using namespace dolfinx;

/// @brief Interpolate a function into a Lagrange finite element space
/// and outputs the finite element function to a VTX file for
/// visualisation.
///
/// @tparam T Scalar type of the finite element function.
/// @tparam U Float type for the finite element basis and the mesh.
/// @param mesh Mesh.
/// @param filename Output filename. File output requires DOLFINX to be
/// configured with ADIOS2.
template <typename T, std::floating_point U>
void interpolate_scalar(std::shared_ptr<mesh::Mesh<U>> mesh,
                        [[maybe_unused]] std::filesystem::path filename)
{
  // Create a Basix continuous Lagrange element of degree 1
  basix::FiniteElement e = basix::create_element<U>(
      basix::element::family::P,
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  // Create a scalar function space
  auto V = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
      mesh, std::make_shared<fem::FiniteElement<U>>(e)));

  // Create a finite element Function
  auto u = std::make_shared<fem::Function<T>>(V);

  // Interpolate sin(2 \pi x[0]) in the scalar Lagrange finite element
  // space
  u->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f(x.extent(1));
        for (std::size_t p = 0; p < x.extent(1); ++p)
          f[p] = std::sin(2 * std::numbers::pi * x(0, p));
        return {f, {f.size()}};
      });

#ifdef HAS_ADIOS2
  // Write the function to a VTX file for visualisation, e.g. using
  // ParaView
  io::VTXWriter<U> outfile(mesh->comm(), filename.replace_extension("bp"), {u},
                           "BP4");
  outfile.write(0.0);
  outfile.close();
#endif
}

/// @brief Interpolate a function into a H(curl) finite element space.
///
/// To visualise the function, the H(curl) finite element function is
/// interpolated in a discontinuous Lagrange space, which is written to
/// a VTX file for visualisation. This allows exact visualisation of a
/// function in H(curl).
///
/// @tparam T Scalar type of the finite element function.
/// @tparam U Float type for the finite element basis and the mesh.
/// @param mesh Mesh.
/// @param filename Output filename. File output requires DOLFINX to be
/// configured with ADIOS2.
template <typename T, std::floating_point U>
void interpolate_nedelec(std::shared_ptr<mesh::Mesh<U>> mesh,
                         [[maybe_unused]] std::filesystem::path filename)
{
  // Create a Basix Nedelec (first kind) element of degree 2 (dim=6 on
  // triangle)
  basix::FiniteElement e = basix::create_element<U>(
      basix::element::family::N1E,
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 2,
      basix::element::lagrange_variant::legendre,
      basix::element::dpc_variant::unset, false);

  // Create a Nedelec function space
  auto V = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
      mesh, std::make_shared<fem::FiniteElement<U>>(e)));

  // Create a Nedelec finite element Function
  auto u = std::make_shared<fem::Function<T>>(V);

  // Interpolate the vector field
  //  u = [x[0], x[1]]  if x[0] < 0.5
  //      [x[0] + 1, x[1]]  if x[0] >= 0.5
  // in the Nedelec space.
  //
  // Note that the x1 component of this field is continuous, and the x0
  // component is discontinuous across x0 = 0.5. This function lies in
  // the Nedelec space when there are cell edges aligned to x0 = 0.5.

  // Find cells with all vertices satisfying (0) x0 <= 0.5 and (1) x0 >= 0.5
  std::vector cells0
      = mesh::locate_entities(*mesh, 2,
                              [](auto x)
                              {
                                std::vector<std::int8_t> marked;
                                for (std::size_t i = 0; i < x.extent(1); ++i)
                                  marked.push_back(x(0, i) <= 0.5);
                                return marked;
                              });
  std::vector cells1
      = mesh::locate_entities(*mesh, 2,
                              [](auto x)
                              {
                                std::vector<std::int8_t> marked;
                                for (std::size_t i = 0; i < x.extent(1); ++i)
                                  marked.push_back(x(0, i) >= 0.5);
                                return marked;
                              });

  // Interpolation on the two sets of cells
  u->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f(2 * x.extent(1), 0.0);
        std::copy_n(x.data_handle(), f.size(), f.begin());
        return {f, {2, x.extent(1)}};
      },
      cells0);
  u->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f(2 * x.extent(1), 0.0);
        std::copy_n(x.data_handle(), f.size(), f.begin());
        std::ranges::transform(f, f.begin(), [](auto x) { return x + T(1); });
        return {f, {2, x.extent(1)}};
      },
      cells1);

  // Nedelec spaces are not generally supported by visualisation tools.
  // Simply evaluating a Nedelec function at cell vertices can
  // mis-represent the function. However, we can represented a Nedelec
  // function exactly in a discontinuous Lagrange space which we can
  // then visualise. We do this here.

  // First create a degree 2 vector-valued discontinuous Lagrange space
  // (which contains the N2 space):
  basix::FiniteElement e_l = basix::create_element<U>(
      basix::element::family::P,
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 2,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, true);

  // Create a function space
  auto V_l
      = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
          mesh, std::make_shared<fem::FiniteElement<U>>(
                    e_l, std::vector<std::size_t>{2})));

  auto u_l = std::make_shared<fem::Function<T>>(V_l);

  // Interpolate the Nedelec function into the discontinuous Lagrange
  // space:
  u_l->interpolate(*u);

// Output the discontinuous Lagrange space in VTX format. When plotting
// the x0 component the field will appear discontinuous at x0 = 0.5
// (jump in the normal component between cells) and the x1 component
// will appear continuous (continuous tangent component between cells).
#ifdef HAS_ADIOS2
  io::VTXWriter<U> outfile(mesh->comm(), filename.replace_extension("bp"),
                           {u_l}, "BP4");
  outfile.write(0.0);
  outfile.close();
#endif
}

/// @brief This program shows how to interpolate functions into different types
/// of finite element spaces and output the result to file for visualisation.
int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  // The main body of the function is scoped to ensure that all objects
  // that depend on an MPI communicator are destroyed before MPI is
  // finalised at the end of this function.
  {
    //  Create meshes. For what comes later in this demo we need to
    //  ensure that a boundary between cells is located at x0=0.5

    // Create mesh using float for geometry coordinates
    auto mesh0
        = std::make_shared<mesh::Mesh<float>>(mesh::create_rectangle<float>(
            MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {32, 4},
            mesh::CellType::triangle,
            mesh::create_cell_partitioner(mesh::GhostMode::none)));

    // Create mesh using same topology as mesh0, but with different
    // scalar type for geometry
    auto mesh1
        = std::make_shared<mesh::Mesh<double>>(mesh::create_rectangle<double>(
            MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {32, 4},
            mesh::CellType::triangle,
            mesh::create_cell_partitioner(mesh::GhostMode::none)));

    // Interpolate a function in a scalar Lagrange space and output the
    // result to file for visualisation using different types
    interpolate_scalar<float>(mesh0, "u32");
    interpolate_scalar<double>(mesh1, "u64");
    interpolate_scalar<std::complex<float>>(mesh0, "u_complex64");
    interpolate_scalar<std::complex<double>>(mesh1, "u_complex128");

    // Interpolate a function in a H(curl) finite element space, and
    // then interpolate the H(curl) function in a discontinuous Lagrange
    // space for visualisation using different types
    interpolate_nedelec<float>(mesh0, "u_nedelec32");
    interpolate_nedelec<double>(mesh1, "u_nedelec64");
    interpolate_nedelec<std::complex<float>>(mesh0, "u_nedelec_complex64");
    interpolate_nedelec<std::complex<double>>(mesh1, "u_nedelec_complex128");
  }

  MPI_Finalize();
  return 0;
}
