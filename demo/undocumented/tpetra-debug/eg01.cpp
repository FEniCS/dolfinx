
#include <dolfin.h>
#include "Poisson.h"

using namespace dolfin;

// Source term (right-hand side)
class Source : public Expression
{
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = x[0];
  }
};

bool compare_tpetra_petsc_vectors(const TpetraVector& t,
                                  const GenericVector& p)
{
  std::vector<double> g_tpetra;
  std::vector<double> g_petsc;

  t.gather_on_zero(g_tpetra);
  p.gather_on_zero(g_petsc);

  bool err = false;

  int mpi_rank = t.vec()->getMap()->getComm()->getRank();

  if(mpi_rank == 0)
    for (unsigned int i = 0; i != g_tpetra.size(); ++i)
    {
      double diff = std::abs(g_petsc[i] - g_tpetra[i]);
      if (diff > 1e-9)
      {
        err = true;
        std::cout << "Difference at " << i << ": PETSC=" << g_petsc[i]
                  << ", TPETRA=" << g_tpetra[i] << "\n";
      }
    }

  return err;
}

int main()
{
  bool dump_matrix = false;

  UnitSquareMesh mesh(2, 5);
  Poisson::FunctionSpace V(mesh);
  Poisson::LinearForm L(V);
  Poisson::BilinearForm a(V, V);

  std::size_t rank = MPI::rank(mesh.mpi_comm());

  std::shared_ptr<FunctionSpace> Vptr(new Poisson::FunctionSpace(mesh));
  std::shared_ptr<TpetraVector> TVptr(new TpetraVector());

  TpetraVector tpetraB;
  PETScVector petscB;

  TpetraMatrix tpetraA;
  PETScMatrix petscA;

  Source f;
  L.f = f;

  assemble(tpetraA, a);
  assemble(petscA, a);

  tpetraA.init_vector(*TVptr, 1);

  std::cout << rank << "] " << TVptr->local_range().first << " - " << TVptr->local_range().second << "\n";

  Function F2(Vptr, TVptr);
  F2.interpolate(f);

  std::cout << "F2.sum() = " << F2.vector()->sum() << "\n";

  TpetraMatrix::graphdump(tpetraA.mat()->getCrsGraph());

  TpetraVector::mapdump(tpetraA.mat()->getDomainMap(), "domain");
  TpetraVector::mapdump(tpetraA.mat()->getRangeMap(), "range");
  TpetraVector::mapdump(tpetraA.mat()->getRowMap(), "row");
  TpetraVector::mapdump(tpetraA.mat()->getColMap(), "col");

  assemble(tpetraB, L);
  assemble(petscB, L);

  compare_tpetra_petsc_vectors(tpetraB, petscB);

  std::cout << "Check transpmult operation ----------------\n";
  TpetraVector tpetraX;
  PETScVector petscX;
  tpetraA.transpmult(tpetraB, tpetraX);
  petscA.transpmult(petscB, petscX);
  tpetraX.apply("add");
  compare_tpetra_petsc_vectors(tpetraX, petscX);

  std::cout << "Check mult operation ----------------\n";
  TpetraVector tpetraY;
  PETScVector petscY;
  tpetraA.transpmult(tpetraB, tpetraY);
  petscA.mult(petscB, petscY);

  tpetraY.apply("add");
  compare_tpetra_petsc_vectors(tpetraY, petscY);

  std::cout << "Check max, min, sum --------------------\n";

  double diff = std::abs(petscB.max() - tpetraB.max());
  if (diff < 1e-12)
    std::cout << "Max OK\n";
  else
    error("MAX");

  diff = std::abs(petscB.min() - tpetraB.min());
  if (diff < 1e-12)
    std::cout << "Min OK\n";
  else
    error("MIN");

  diff = std::abs(petscB.sum() - tpetraB.sum());
  if (diff < 1e-12)
    std::cout << "Sum OK\n";
  else
    error("SUM");

  if (dump_matrix)
  {
    std::cout << "Matrix properties ----------------------\n";

    std::cout << tpetraA.str(true) << "\n";
    std::pair<std::size_t, std::size_t> range1 = tpetraA.local_range(1);
    std::cout << "r(1) = " << range1.first << " - " << range1.second << "\n";
    std::pair<std::size_t, std::size_t> range = petscA.local_range(0);
    std::cout << "r(0) = " << range.first << " - " << range.second << "\n";

    std::stringstream ss;

    Teuchos::RCP<matrix_type> m = tpetraA.mat();

    for (std::size_t i = 0; i != m->getRowMap()->getNodeNumElements(); ++i)
    {
      std::vector<double> data;
      std::vector<std::size_t> cols;
      std::size_t gi = m->getRowMap()->getGlobalElement(i);

      tpetraA.getrow(gi, cols, data);

      ss << gi << "] ";
      for (std::size_t j = 0; j != data.size(); ++j)
        ss << "(" << cols[j] << ", " << data[j] << ") ";
      ss << "\n";

    }

    for (std::size_t i = range.first; i != range.second; ++i)
    {
      std::vector<double> data;
      std::vector<std::size_t> cols;
      petscA.getrow(i, cols, data);

      ss << "P "<<  i << "] ";
      for (std::size_t j = 0; j != data.size(); ++j)
        ss << "(" << cols[j] << ", " << data[j] << ") ";
      ss << "\n";
    }

    std::cout << ss.str();
  }

  return 0;
}
