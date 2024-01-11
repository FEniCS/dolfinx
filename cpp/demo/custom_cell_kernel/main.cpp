// Custom cell kernel (C++)
// .. code-block:: cpp

#include <cmath>
#include <dolfinx.h>
#include <utility>
#include <vector>

using namespace dolfinx;

using T = double;
using U = typename dolfinx::scalar_value_type_t<T>;

// .. code-block:: cpp

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {32, 16}, mesh::CellType::triangle, part));

    //auto V = std::make_shared<fem::FunctionSpace<U>>(
    //    fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

    // Define variational forms
    //auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
    //    *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
  }

  return 0;
}
