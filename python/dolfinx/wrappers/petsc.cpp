
#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include "pycoeff.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/discreteoperators.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/nls/NewtonSolver.h>

#include <petsc4py/petsc4py.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace
{

// Declare assembler function that have multiple scalar types
template <typename T, typename U>
void declare_petsc_discrete_operators(py::module& m)
{
  m.def(
      "discrete_gradient",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        auto dofmap0 = V0.dofmap();
        assert(dofmap0);
        auto dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

        int tdim = mesh->topology()->dim();
        auto map = mesh->topology()->index_map(tdim);
        assert(map);
        std::vector<std::int32_t> c(map->size_local(), 0);
        std::iota(c.begin(), c.end(), 0);
        dolfinx::fem::sparsitybuild::cells(sp, c, {*dofmap1, *dofmap0});
        sp.finalize();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::discrete_gradient<T, U>(
            *V0.mesh()->topology_mutable(), {*V0.element(), *V0.dofmap()},
            {*V1.element(), *V1.dofmap()},
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES));
        return A;
      },
      py::return_value_policy::take_ownership, py::arg("V0"), py::arg("V1"));

  m.def(
      "interpolation_matrix",
      [](const dolfinx::fem::FunctionSpace<U>& V0,
         const dolfinx::fem::FunctionSpace<U>& V1)
      {
        assert(V0.mesh());
        auto mesh = V0.mesh();
        assert(V1.mesh());
        assert(mesh == V1.mesh());
        MPI_Comm comm = mesh->comm();

        auto dofmap0 = V0.dofmap();
        assert(dofmap0);
        auto dofmap1 = V1.dofmap();
        assert(dofmap1);

        // Create and build  sparsity pattern
        assert(dofmap0->index_map);
        assert(dofmap1->index_map);
        dolfinx::la::SparsityPattern sp(
            comm, {dofmap1->index_map, dofmap0->index_map},
            {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

        int tdim = mesh->topology()->dim();
        auto map = mesh->topology()->index_map(tdim);
        assert(map);
        std::vector<std::int32_t> c(map->size_local(), 0);
        std::iota(c.begin(), c.end(), 0);
        dolfinx::fem::sparsitybuild::cells(sp, c, {*dofmap1, *dofmap0});
        sp.finalize();

        // Build operator
        Mat A = dolfinx::la::petsc::create_matrix(comm, sp);
        MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
        dolfinx::fem::interpolation_matrix<T, U>(
            V0, V1, dolfinx::la::petsc::Matrix::set_block_fn(A, INSERT_VALUES));
        return A;
      },
      py::return_value_policy::take_ownership, py::arg("V0"), py::arg("V1"));
}

void petsc_la_module(py::module& m)
{
  m.def("create_vector",
        py::overload_cast<const dolfinx::common::IndexMap&, int>(
            &dolfinx::la::petsc::create_vector),
        py::return_value_policy::take_ownership, py::arg("index_map"),
        py::arg("bs"), "Create a ghosted PETSc Vec for index map.");
  m.def(
      "create_vector_wrap",
      [](dolfinx::la::Vector<PetscScalar>& x)
      { return dolfinx::la::petsc::create_vector_wrap(x); },
      py::return_value_policy::take_ownership, py::arg("x"),
      "Create a ghosted PETSc Vec that wraps a DOLFINx Vector");
  m.def(
      "create_matrix",
      [](dolfinx_wrappers::MPICommWrapper comm,
         const dolfinx::la::SparsityPattern& p, const std::string& type)
      { return dolfinx::la::petsc::create_matrix(comm.get(), p, type); },
      py::return_value_policy::take_ownership, py::arg("comm"), py::arg("p"),
      py::arg("type") = std::string(),
      "Create a PETSc Mat from sparsity pattern.");

  // TODO: check reference counting for index sets
  m.def("create_index_sets", &dolfinx::la::petsc::create_index_sets,
        py::arg("maps"), py::return_value_policy::take_ownership);

  m.def(
      "scatter_local_vectors",
      [](Vec x,
         const std::vector<py::array_t<PetscScalar, py::array::c_style>>& x_b,
         const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps)
      {
        std::vector<std::span<const PetscScalar>> _x_b;
        for (auto& array : x_b)
          _x_b.emplace_back(array.data(), array.size());
        dolfinx::la::petsc::scatter_local_vectors(x, _x_b, maps);
      },
      py::arg("x"), py::arg("x_b"), py::arg("maps"),
      "Scatter the (ordered) list of sub vectors into a block "
      "vector.");
  m.def(
      "get_local_vectors",
      [](const Vec x,
         const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps)
      {
        std::vector<std::vector<PetscScalar>> vecs
            = dolfinx::la::petsc::get_local_vectors(x, maps);
        std::vector<py::array> ret;
        for (std::vector<PetscScalar>& v : vecs)
          ret.push_back(dolfinx_wrappers::as_pyarray(std::move(v)));
        return ret;
      },
      py::arg("x"), py::arg("maps"),
      "Gather an (ordered) list of sub vectors from a block vector.");
}

void petsc_fem_module(py::module& m)
{
  // Create PETSc vectors and matrices
  m.def("create_vector_block", &dolfinx::fem::petsc::create_vector_block,
        py::return_value_policy::take_ownership, py::arg("maps"),
        "Create a monolithic vector for multiple (stacked) linear forms.");
  m.def("create_vector_nest", &dolfinx::fem::petsc::create_vector_nest,
        py::return_value_policy::take_ownership, py::arg("maps"),
        "Create nested vector for multiple (stacked) linear forms.");
  m.def("create_matrix", dolfinx::fem::petsc::create_matrix<PetscReal>,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("type") = std::string(),
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block",
        &dolfinx::fem::petsc::create_matrix_block<PetscReal>,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("type") = std::string(),
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest",
        &dolfinx::fem::petsc::create_matrix_nest<PetscReal>,
        py::return_value_policy::take_ownership, py::arg("a"),
        py::arg("types") = std::vector<std::vector<std::string>>(),
        "Create nested sparse matrix for bilinear forms.");

  // PETSc Matrices
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<PetscScalar, PetscReal>& a,
         const py::array_t<PetscScalar, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<PetscScalar, py::array::c_style>>&
             coefficients,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar, PetscReal>>>& bcs,
         bool unrolled)
      {
        if (unrolled)
        {
          auto set_fn = dolfinx::la::petsc::Matrix::set_block_expand_fn(
              A, a.function_spaces()[0]->dofmap()->bs(),
              a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES);
          dolfinx::fem::assemble_matrix(
              set_fn, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
        else
        {
          dolfinx::fem::assemble_matrix(
              dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), a,
              std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), bcs);
        }
      },
      py::arg("A"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("bcs"), py::arg("unrolled") = false,
      "Assemble bilinear form into an existing PETSc matrix");
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<PetscScalar, PetscReal>& a,
         const py::array_t<PetscScalar, py::array::c_style>& constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        py::array_t<PetscScalar, py::array::c_style>>&
             coefficients,
         const py::array_t<std::int8_t, py::array::c_style>& rows0,
         const py::array_t<std::int8_t, py::array::c_style>& rows1,
         bool unrolled)
      {
        if (rows0.ndim() != 1 or rows1.ndim())
        {
          throw std::runtime_error(
              "Expected 1D arrays for boundary condition rows/columns");
        }

        std::function<int(const std::span<const std::int32_t>&,
                          const std::span<const std::int32_t>&,
                          const std::span<const PetscScalar>&)>
            set_fn;
        if (unrolled)
        {
          set_fn = dolfinx::la::petsc::Matrix::set_block_expand_fn(
              A, a.function_spaces()[0]->dofmap()->bs(),
              a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES);
        }
        else
          set_fn = dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES);

        dolfinx::fem::assemble_matrix(
            set_fn, a, std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients),
            std::span(rows0.data(), rows0.size()),
            std::span(rows1.data(), rows1.size()));
      },
      py::arg("A"), py::arg("a"), py::arg("constants"), py::arg("coeffs"),
      py::arg("rows0"), py::arg("rows1"), py::arg("unrolled") = false);
  m.def(
      "insert_diagonal",
      [](Mat A, const dolfinx::fem::FunctionSpace<PetscReal>& V,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar, PetscReal>>>& bcs,
         PetscScalar diagonal)
      {
        dolfinx::fem::set_diagonal(
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES), V, bcs,
            diagonal);
      },
      py::arg("A"), py::arg("V"), py::arg("bcs"), py::arg("diagonal"));

  declare_petsc_discrete_operators<PetscScalar, PetscReal>(m);
}

void petsc_nls_module(py::module& m)
{
  // dolfinx::NewtonSolver
  py::class_<dolfinx::nls::petsc::NewtonSolver,
             std::shared_ptr<dolfinx::nls::petsc::NewtonSolver>>(m,
                                                                 "NewtonSolver")
      .def(py::init(
               [](const dolfinx_wrappers::MPICommWrapper comm) {
                 return std::make_unique<dolfinx::nls::petsc::NewtonSolver>(
                     comm.get());
               }),
           py::arg("comm"))
      .def_property_readonly("krylov_solver",
                             [](const dolfinx::nls::petsc::NewtonSolver& self)
                             {
                               const dolfinx::la::petsc::KrylovSolver& solver
                                   = self.get_krylov_solver();
                               return solver.ksp();
                             })
      .def("setF", &dolfinx::nls::petsc::NewtonSolver::setF, py::arg("F"),
           py::arg("b"))
      .def("setJ", &dolfinx::nls::petsc::NewtonSolver::setJ, py::arg("J"),
           py::arg("Jmat"))
      .def("setP", &dolfinx::nls::petsc::NewtonSolver::setP, py::arg("P"),
           py::arg("Pmat"))
      .def("set_update", &dolfinx::nls::petsc::NewtonSolver::set_update,
           py::arg("update"))
      .def("set_form", &dolfinx::nls::petsc::NewtonSolver::set_form,
           py::arg("form"))
      .def("solve", &dolfinx::nls::petsc::NewtonSolver::solve, py::arg("x"))
      .def_readwrite("atol", &dolfinx::nls::petsc::NewtonSolver::atol,
                     "Absolute tolerance")
      .def_readwrite("rtol", &dolfinx::nls::petsc::NewtonSolver::rtol,
                     "Relative tolerance")
      .def_readwrite(
          "error_on_nonconvergence",
          &dolfinx::nls::petsc::NewtonSolver::error_on_nonconvergence)
      .def_readwrite("report", &dolfinx::nls::petsc::NewtonSolver::report)
      .def_readwrite("relaxation_parameter",
                     &dolfinx::nls::petsc::NewtonSolver::relaxation_parameter,
                     "Relaxation parameter")
      .def_readwrite("max_it", &dolfinx::nls::petsc::NewtonSolver::max_it,
                     "Maximum number of iterations")
      .def_readwrite("convergence_criterion",
                     &dolfinx::nls::petsc::NewtonSolver::convergence_criterion,
                     "Convergence criterion, either 'residual' (default) or "
                     "'incremental'");
}

} // namespace

namespace dolfinx_wrappers
{

void petsc(py::module& m_fem, py::module& m_la, py::module& m_nls)
{
  py::module petsc_fem_mod
      = m_fem.def_submodule("petsc", "PETSc-specific finite element module");
  petsc_fem_module(petsc_fem_mod);

  py::module petsc_la_mod
      = m_la.def_submodule("petsc", "PETSc-specific linear algebra module");
  petsc_la_module(petsc_la_mod);

  py::module petsc_nls_mod
      = m_nls.def_submodule("petsc", "PETSc-specific non-linear solvers");
  petsc_nls_module(petsc_nls_mod);
}
} // namespace dolfinx_wrappers
