
#include <dolfin.h>
// #include <Amesos2.hpp>

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


int main()
{
  UnitSquareMesh mesh(2, 5);
  Poisson::FunctionSpace V(mesh);
  Poisson::LinearForm L(V);
  Poisson::BilinearForm a(V, V);

  TpetraVector t, t2;
  PETScVector tp, tp2;

  TpetraMatrix mat;
  PETScMatrix matp;

  Source f;
  L.f = f;

  assemble(mat, a);
  assemble(matp, a);

  typedef Tpetra::CrsGraph<> graph_type;
  Teuchos::RCP<const graph_type> g = mat.mat()->getCrsGraph();
  std::stringstream ss;

  ss << g->description();
  ss << "\n Rank" << MPI::rank(mat.mpi_comm()) << "\n";
  for(int i=0; i!= g->getRowMap()->getNodeNumElements(); ++i)
  {
    global_ordinal_type idx = g->getRowMap()->getGlobalElement(i);

    const int nx = g->getNumEntriesInGlobalRow(idx);
    std::vector<global_ordinal_type> rowvec(nx);
    Teuchos::ArrayView<global_ordinal_type> _rowvec(rowvec);
    std::size_t n;
    g->getGlobalRowCopy(idx, _rowvec, n);
    ss << idx << "] ";
    for (int j=0; j !=nx; ++j)
      ss << rowvec[j] << " ";
    ss<< "\n";
  }
  std::cout << ss.str();


  if (true)
  {

  TpetraVector::mapdump(mat.mat()->getDomainMap(), "domain");
  TpetraVector::mapdump(mat.mat()->getRangeMap(), "range");
  TpetraVector::mapdump(mat.mat()->getRowMap(), "row");
  TpetraVector::mapdump(mat.mat()->getColMap(), "col");

  assemble(t, L);
  assemble(tp, L);

  mat.transpmult(t, t2);
  matp.transpmult(tp, tp2);

  //  Teuchos::RCP<Amesos2::Solver<matrix_type, vector_type> > solver 
  //    = Amesos2::create<matrix_type, vector_type> ("Superlu", mat.mat(), 
  //                                                 t.vec(), t2.vec());

  //  solver->solve();

  std::cout << "VECTOR: " << t.str(true) << "\n";

  std::cout << "t.size() = " << t.size() << "\n";

  std::cout << mat.str(true) << "\n";

  std::pair<std::size_t, std::size_t> range1 = mat.local_range(1);
  std::cout << "r(1) = " << range1.first << " - " << range1.second << "\n";
  std::pair<std::size_t, std::size_t> range = matp.local_range(0);
  std::cout << "r(0) = " << range.first << " - " << range.second << "\n";

  std::stringstream ss;

  Teuchos::RCP<matrix_type> m = mat.mat();

  for (std::size_t i = 0; i != m->getRowMap()->getNodeNumElements(); ++i)
  {
    std::vector<double> data;
    std::vector<std::size_t> cols;
    std::size_t gi = m->getRowMap()->getGlobalElement(i);

    mat.getrow(gi, cols, data);

    ss << gi << "] ";
    for (std::size_t j = 0; j != data.size(); ++j)
      ss << "(" << cols[j] << ", " << data[j] << ") ";
     ss << "\n";

  }

  for (std::size_t i = range.first; i != range.second; ++i)
  {
    std::vector<double> data;
    std::vector<std::size_t> cols;
    matp.getrow(i, cols, data);

    ss << "P "<<  i << "] ";
    for (std::size_t j = 0; j != data.size(); ++j)
      ss << "(" << cols[j] << ", " << data[j] << ") ";
    ss << "\n";
  }

  std::cout << ss.str();

  std::cout << "t.max() = " << t.max() << "\n";

  std::cout << t.str(true) << ", local_size - " << t.local_size() << ": ";
  std::cout << t.local_range().first << " - " << t.local_range().second << "\n";

  std::cout << "t.sum() = " << t.sum() < "\n";

  ss.str("");

  ss << "[" << MPI::rank(mesh.mpi_comm()) << "] ";
  for (unsigned int i = 0; i != t.size(); ++i)
    ss << t.owns_index(i);

  ss << " " << t.local_range().first << "-" << t.local_range().second << "\n";

  std::cout << ss.str();
  ss.str("");

  for (int i = 0; i != (int)t.size(); ++i)
  {
    double val = 0;
    t.get(&val, 1, &i);
    if (t.owns_index(i))
    {
      ss << "Tpetra: " << i << ") " << val << "\n";
    }

  }

  std::cout << ss.str();
  ss.str("");

  for (int i = 0; i != (int)tp.size(); ++i)
  {
    double val = 0;
    if (tp.owns_index(i))
    {
      tp.get(&val, 1, &i);
      ss << "PETSC: " << i << ") " << val << "\n";
    }

  }

  std::cout << ss.str() << "\n";

  std::vector<double> g_tpetra;
  std::vector<double> g_petsc;

  t2.gather_on_zero(g_tpetra);
  tp2.gather_on_zero(g_petsc);

  if(MPI::rank(mesh.mpi_comm()) == 0)
    for (unsigned int i = 0; i != g_tpetra.size(); ++i)
      std::cout << g_tpetra[i] << "/ P" << g_petsc[i] << " / ";
  }

  return 0;
}
