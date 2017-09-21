// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "casters.h"

#include <dolfin/common/Array.h>
#include <dolfin/la/solve.h>
#include <dolfin/la/BlockVector.h>
#include <dolfin/la/BlockMatrix.h>
#include <dolfin/la/GenericLinearOperator.h>
#include <dolfin/la/GenericLinearSolver.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/IndexMap.h>
#include <dolfin/la/LinearAlgebraObject.h>
#include <dolfin/la/LinearOperator.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/la/EigenFactory.h>
#include <dolfin/la/EigenMatrix.h>
#include <dolfin/la/EigenVector.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScLUSolver.h>
#include <dolfin/la/PETScFactory.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScOptions.h>
#include <dolfin/la/PETScPreconditioner.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/TpetraFactory.h>
#include <dolfin/la/TpetraMatrix.h>
#include <dolfin/la/TpetraVector.h>
#include <dolfin/la/TrilinosPreconditioner.h>
#include <dolfin/la/MueluPreconditioner.h>
#include <dolfin/la/BelosKrylovSolver.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/la/SLEPcEigenSolver.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/la/solve.h>
#include <dolfin/la/VectorSpaceBasis.h>
#include <dolfin/la/test_nullspace.h>

namespace py = pybind11;

namespace
{
  template<typename T>
  void check_indices(const py::array_t<T>& x, std::int64_t local_size)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      std::int64_t _x = *(x.data() + i);
      if (_x < 0 or !(_x < local_size))
        throw py::index_error("Vector index out of range");
    }
  }
}

namespace dolfin_wrappers
{

  using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  void la(py::module& m)
  {
#ifdef HAS_PETSC4PY
    int ierr = import_petsc4py();
    if (ierr != 0)
      throw std::runtime_error("Failed to import petsc4py");
#endif

    // dolfin::IndexMap
    py::class_<dolfin::IndexMap, std::shared_ptr<dolfin::IndexMap>> index_map(m, "IndexMap");
    index_map.def("size", &dolfin::IndexMap::size);
    index_map.def("local_range", &dolfin::IndexMap::local_range);

    // dolfin::IndexMap enums
    py::enum_<dolfin::IndexMap::MapSize>(index_map, "MapSize")
      .value("ALL", dolfin::IndexMap::MapSize::ALL)
      .value("OWNED", dolfin::IndexMap::MapSize::OWNED)
      .value("UNOWNED", dolfin::IndexMap::MapSize::UNOWNED)
      .value("GLOBAL", dolfin::IndexMap::MapSize::GLOBAL);

    // dolfin::SparsityPattern
    py::class_<dolfin::SparsityPattern, std::shared_ptr<dolfin::SparsityPattern>>(m, "SparsityPattern")
      .def("init", &dolfin::SparsityPattern::init)
      .def("apply", &dolfin::SparsityPattern::apply)
      .def("str", &dolfin::SparsityPattern::str)
      .def("num_nonzeros", &dolfin::SparsityPattern::num_nonzeros)
      .def("num_nonzeros_diagonal", [](const dolfin::SparsityPattern& instance)
           {
             std::vector<std::size_t> num_nonzeros;
             instance.num_nonzeros_diagonal(num_nonzeros);
             return py::array_t<std::size_t>(num_nonzeros.size(), num_nonzeros.data());
           })
      .def("num_nonzeros_off_diagonal", [](const dolfin::SparsityPattern& instance)
           {
             std::vector<std::size_t> num_nonzeros;
             instance.num_nonzeros_off_diagonal(num_nonzeros);
             return py::array_t<std::size_t>(num_nonzeros.size(), num_nonzeros.data());
           })
      .def("num_local_nonzeros", [](const dolfin::SparsityPattern& instance)
           {
             std::vector<std::size_t> num_nonzeros;
             instance.num_local_nonzeros(num_nonzeros);
             return py::array_t<std::size_t>(num_nonzeros.size(), num_nonzeros.data());
           })
      // FIXME: Switch EigenMap in DOLFIN interface when SWIG is dropped
      .def("insert_local", [](dolfin::SparsityPattern& self,
                              std::vector<Eigen::Matrix<dolfin::la_index, Eigen::Dynamic, 1>> entries)
           {
             std::vector<dolfin::ArrayView<const dolfin::la_index>> e(entries.size());
             for (std::size_t i = 0; i < entries.size(); ++i)
               e[i] = dolfin::ArrayView<const dolfin::la_index>(entries[i].size(), &entries[i][0]);

             self.insert_local(e);
           })
      .def("insert_global", [](dolfin::SparsityPattern& self,
                              std::vector<Eigen::Matrix<dolfin::la_index, Eigen::Dynamic, 1>> entries)
           {
             std::vector<dolfin::ArrayView<const dolfin::la_index>> e(entries.size());
             for (std::size_t i = 0; i < entries.size(); ++i)
               e[i] = dolfin::ArrayView<const dolfin::la_index>(entries[i].size(), &entries[i][0]);

             self.insert_global(e);
           })
      .def("insert_local_global", [](dolfin::SparsityPattern& self,
                                     std::vector<Eigen::Matrix<dolfin::la_index, Eigen::Dynamic, 1>> entries)
           {
             std::vector<dolfin::ArrayView<const dolfin::la_index>> e(entries.size());
             for (std::size_t i = 0; i < entries.size(); ++i)
               e[i] = dolfin::ArrayView<const dolfin::la_index>(entries[i].size(), &entries[i][0]);

             self.insert_local_global(e);
           });

    // dolfin::TensorLayout
    py::class_<dolfin::TensorLayout, std::shared_ptr<dolfin::TensorLayout>> tensor_layout(m, "TensorLayout");

    // dolfin::TensorLayout enums
    py::enum_<dolfin::TensorLayout::Sparsity>(tensor_layout, "Sparsity")
      .value("SPARSE", dolfin::TensorLayout::Sparsity::SPARSE)
      .value("DENSE", dolfin::TensorLayout::Sparsity::DENSE);
    py::enum_<dolfin::TensorLayout::Ghosts>(tensor_layout, "Ghosts")
      .value("GHOSTED", dolfin::TensorLayout::Ghosts::GHOSTED)
      .value("UNGHOSTED", dolfin::TensorLayout::Ghosts::UNGHOSTED);

    tensor_layout
      .def(py::init<MPI_Comm, std::size_t, dolfin::TensorLayout::Sparsity>())
      .def(py::init<MPI_Comm, std::vector<std::shared_ptr<const dolfin::IndexMap>>,
           std::size_t, dolfin::TensorLayout::Sparsity, dolfin::TensorLayout::Ghosts>())
      .def("init", &dolfin::TensorLayout::init)
      .def("sparsity_pattern", (std::shared_ptr<dolfin::SparsityPattern> (dolfin::TensorLayout::*)()) &dolfin::TensorLayout::sparsity_pattern);

    // dolfin::LinearAlgebraObject
    py::class_<dolfin::LinearAlgebraObject, std::shared_ptr<dolfin::LinearAlgebraObject>,
               dolfin::Variable>(m, "LinearAlgebraObject")
    .def("mpi_comm", &dolfin::GenericLinearOperator::mpi_comm);

    // dolfin::GenericLinearOperator
    py::class_<dolfin::GenericLinearOperator, std::shared_ptr<dolfin::GenericLinearOperator>,
               dolfin::LinearAlgebraObject>
      (m, "GenericLinearOperator", "DOLFIN GenericLinearOperator object")
      .def("mult", &dolfin::GenericLinearOperator::mult);

    // dolfin::GenericTensor
    py::class_<dolfin::GenericTensor, std::shared_ptr<dolfin::GenericTensor>,
               dolfin::LinearAlgebraObject>
      (m, "GenericTensor", "DOLFIN GenericTensor object")
      .def("init", &dolfin::GenericTensor::init)
      .def("zero", &dolfin::GenericTensor::zero);

    // dolfin::GenericMatrix
    py::class_<dolfin::GenericMatrix, std::shared_ptr<dolfin::GenericMatrix>,
               dolfin::GenericTensor, dolfin::GenericLinearOperator>
      (m, "GenericMatrix", "DOLFIN GenericMatrix object")
      .def("init_vector", &dolfin::GenericMatrix::init_vector)
      .def("axpy", &dolfin::GenericMatrix::axpy)
      .def("transpmult", &dolfin::GenericMatrix::transpmult)
      // __ifoo__
      .def("__imul__", &dolfin::GenericMatrix::operator*=, "Multiply by a scalar")
      .def("__itruediv__", &dolfin::GenericMatrix::operator/=, py::is_operator(), "Divide by a scalar")
      // Below is an examle of a hand-wrapped in-place operator. Note
      // the explicit return type (const reference). This is necessary
      // to avoid segfaults. Need to investigate more how pybind11
      // handles return types for operators.
      //.def("__itruediv__", [](dolfin::GenericMatrix& self, double a) -> const dolfin::GenericMatrix&
      //     {
      //       self /= a;
      //       return self;
      //     }, py::is_operator(), "Divide by a scalar")
      .def("__iadd__", &dolfin::GenericMatrix::operator+=, py::is_operator(), "Add Matrix")
      .def("__isub__", &dolfin::GenericMatrix::operator-=, py::is_operator(), "Subtract Matrix")
      // __add__
      .def("__add__", [](const dolfin::GenericMatrix& self, const dolfin::GenericMatrix& B)
           { auto C = self.copy(); (*C) += B; return C; }, py::is_operator())
      // __sub__
      .def("__sub__", [](const dolfin::GenericMatrix& self, const dolfin::GenericMatrix& B)
           { auto C = self.copy(); (*C) -= B; return C; }, py::is_operator())
      // __mul__
      .def("__mul__", [](const dolfin::GenericMatrix& self, double a)
           { auto B = self.copy(); (*B) *= a; return B; }, py::is_operator())
      .def("__rmul__", [](const dolfin::GenericMatrix& self, double a)
           { auto B = self.copy(); (*B) *= a; return B; }, py::is_operator())
      .def("__mul__", [](const dolfin::GenericMatrix& self, const dolfin::GenericVector& x)
           {
             auto y = x.factory().create_vector(x.mpi_comm());
             self.init_vector(*y, 0);
             self.mult(x, *y);
             return y;
           }, py::is_operator())
      .def("__mul__", [](const dolfin::GenericMatrix& self, const py::array_t<double> x)
           {
             if (x.ndim() != 1)
               throw py::index_error("NumPy must be a 1D array for multiplication by a GenericMatrix");
             if (x.size() != self.size(1))
               throw py::index_error("Length of array must match number of matrix columns");

             auto _x = self.factory().create_vector(self.mpi_comm());
             self.init_vector(*_x, 1);
             std::vector<double> values(x.data(), x.data() + x.size());
             _x->set_local(values);
             _x->apply("insert");

             auto y = self.factory().create_vector(self.mpi_comm());
             self.init_vector(*y, 0);

             self.mult(*_x, *y);

             y->get_local(values);
             return py::array_t<double>(values.size(), values.data());
           }, "Multiply a DOLFIN matrix and a NumPy array (non-distributed matricds only)")
      // __div__
      .def("__truediv__", [](const dolfin::GenericMatrix& self, double a)
           { auto B = self.copy(); (*B) /= a; return B; }, py::is_operator())
      //
      .def("copy", &dolfin::GenericMatrix::copy)
      .def("local_range", &dolfin::GenericMatrix::local_range)
      .def("norm", &dolfin::GenericMatrix::norm)
      .def("nnz", &dolfin::GenericMatrix::nnz)
      .def("size", &dolfin::GenericMatrix::size)
      .def("apply", &dolfin::GenericMatrix::apply)
      .def("get_diagonal", &dolfin::GenericMatrix::get_diagonal)
      .def("set_diagonal", &dolfin::GenericMatrix::set_diagonal)
      .def("ident_zeros", &dolfin::GenericMatrix::ident_zeros)
      .def("ident", [](dolfin::GenericMatrix& self, std::vector<dolfin::la_index> rows)
           { self.ident(rows.size(), rows.data()); }, py::arg("rows"))
      .def("get", [](dolfin::GenericMatrix& self, Eigen::Ref<RowMatrixXd> block,
                     const std::vector<dolfin::la_index> rows,
                     const std::vector<dolfin::la_index> cols)
           {
             if (block.rows() != rows.size())
               throw py::value_error("Block must have the same number of rows as len(rows)");
             if (block.cols() != cols.size())
               throw py::value_error("Block must have the same number of columns as len(cols)");
             self.get((double *) block.data(), rows.size(), rows.data(),
                      cols.size(), cols.data());
           }, py::arg("block"), py::arg("rows"), py::arg("cols"))
      .def("set", [](dolfin::GenericMatrix& self, const Eigen::Ref<const RowMatrixXd> block,
                     const std::vector<dolfin::la_index> rows,
                     const std::vector<dolfin::la_index> cols)
           {
             if (block.rows() != rows.size())
               throw py::value_error("Block must have the same number of rows as len(rows)");
             if (block.cols() != cols.size())
               throw py::value_error("Block must have the same number of columns as len(cols)");
             self.set((const double *) block.data(), rows.size(), rows.data(),
                      cols.size(), cols.data());
           }, py::arg("block"), py::arg("rows"), py::arg("cols"))
      .def("getrow", [](const dolfin::GenericMatrix& instance, std::size_t row)
           {
             std::vector<double> values;
             std::vector<std::size_t> columns;
             instance.getrow(row, columns, values);
             auto _columns = py::array_t<std::size_t>(columns.size(), columns.data());
             auto _values = py::array_t<double>(values.size(), values.data());
             return std::make_pair(_columns, _values);
           }, py::arg("row"))
      .def("array", [](const dolfin::GenericMatrix& instance)
           {
             // FIXME: This function is highly dubious. It assumes a
             // particular matrix data layout.

             auto m_range = instance.local_range(0);
             std::size_t num_rows = m_range.second - m_range.first;
             std::size_t num_cols = instance.size(1);

             Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_rows, num_cols);
             std::vector<std::size_t> columns;
             std::vector<double> values;
             for (std::size_t i = 0; i < num_rows; ++i)
             {
               const std::size_t row = i + m_range.first;
               instance.getrow(row, columns, values);
               for (std::size_t j = 0; j < columns.size(); ++j)
                 A(i, columns[j]) = values[j];
             }

             return A;
           });

    // dolfin::GenericVector
    py::class_<dolfin::GenericVector, std::shared_ptr<dolfin::GenericVector>,
               dolfin::GenericTensor>
      (m, "GenericVector", "DOLFIN GenericVector object")
      .def("init", (void (dolfin::GenericVector::*)(std::size_t)) &dolfin::GenericVector::init)
      .def("init", (void (dolfin::GenericVector::*)(const dolfin::TensorLayout&)) &dolfin::GenericVector::init)
      .def("init", (void (dolfin::GenericVector::*)(std::pair<std::size_t, std::size_t>)) &dolfin::GenericVector::init)
      .def("copy", &dolfin::GenericVector::copy)
      // sub
      .def("__isub__", (const dolfin::GenericVector& (dolfin::GenericVector::*)(double))
           &dolfin::GenericVector::operator-=)
      .def("__isub__", (const dolfin::GenericVector& (dolfin::GenericVector::*)(const dolfin::GenericVector&))
           &dolfin::GenericVector::operator-=)
      .def("__sub__", [](dolfin::GenericVector& self, double a)
           { auto u = self.copy(); (*u) -= a; return u; }, py::is_operator())
      .def("__sub__", [](dolfin::GenericVector& self, const dolfin::GenericVector& v)
           { auto u = self.copy(); (*u) -= v; return u; }, py::is_operator())
      .def("__rsub__", [](dolfin::GenericVector& self, double a)
           { auto u = self.copy(); (*u) *= -1 ; (*u) += a; return u; }, py::is_operator())
      // div
      .def("__itruediv__", (const dolfin::GenericVector& (dolfin::GenericVector::*)(double))
           &dolfin::GenericVector::operator/=)
      .def("__truediv__", [](const dolfin::GenericVector& instance, double a)
           { auto x = instance.copy(); *x /= a; return x; })
      // add
      .def("__iadd__", (const dolfin::GenericVector& (dolfin::GenericVector::*)(double))
           &dolfin::GenericVector::operator+=)
      .def("__iadd__", (const dolfin::GenericVector& (dolfin::GenericVector::*)(const dolfin::GenericVector&))
           &dolfin::GenericVector::operator+=)
      .def("__add__", [](const dolfin::GenericVector& self, double a)
           { auto x = self.copy(); *x += a; return x;} )
      .def("__add__", [](const dolfin::GenericVector& self, const dolfin::GenericVector& x)
           { auto y = self.copy(); *y += x; return y;} )
      .def("__radd__", [](const dolfin::GenericVector& self, double a)
           { auto x = self.copy(); *x += a; return x;} )
      // mult
      .def("__imul__", (const dolfin::GenericVector& (dolfin::GenericVector::*)(double))
           &dolfin::GenericVector::operator*=)
      .def("__imul__", (const dolfin::GenericVector& (dolfin::GenericVector::*)(const dolfin::GenericVector&))
           &dolfin::GenericVector::operator*=)
      .def("__mul__", [](dolfin::GenericVector& v, double a)
           { auto u = v.copy(); *u *= a; return u; })
      .def("__mul__", [](const dolfin::GenericVector& v, const dolfin::GenericVector& u)
           { auto w = v.copy(); (*w) *= u; return w; }, "Component-wise multiplication of two vectors")
      .def("__rmul__", [](dolfin::GenericVector& v, double a)
           { auto u = v.copy(); *u *= a; return u; })
      // __getitem___
      .def("__getitem__", [](dolfin::GenericVector& self, py::slice slice)
           {
             std::size_t start, stop, step, slicelength;
             if (!slice.compute(self.local_size(), &start, &stop, &step, &slicelength))
               throw py::error_already_set();

             std::vector<double> values(slicelength);
             if (start != 0 or stop != self.local_size() or step != 1)
             {
               std::vector<dolfin::la_index> indices(slicelength, start);
               for (size_t i = 1; i < slicelength; ++i)
                 indices[i] = indices[i-1] + step;
               self.get_local(values.data(), values.size(), indices.data());
             }
             else
               self.get_local(values);

             return py::array_t<double>(values.size(), values.data());
           })
      .def("__getitem__", [](const dolfin::GenericVector& self, py::array_t<bool> indices)
           {
             if (indices.ndim() != 1)
               throw py::index_error("Indices must be a 1D array");
             if (indices.size() != self.local_size())
               throw py::index_error("Indices size mismatch");
             check_indices(indices, self.local_size());

             // Get the values
             std::vector<double> values;
             self.get_local(values);

             // Extract filtered values
             std::vector<double> filtered;
             for (std::size_t i = 0; i < indices.size(); ++i)
             {
               bool e = *(indices.data() + i);
               if (e)
                 filtered.push_back(values[i]);
             }

             return py::array_t<double>(filtered.size(), filtered.data());
           }, py::arg().noconvert())  // Use noconvert to avoid integers being converted to bool
      .def("__getitem__", [](dolfin::GenericVector& self, double index)
           { throw py::type_error("Cannot use float for GenericVector indexing with floats"); }, py::arg().noconvert())
      .def("__getitem__", [](dolfin::GenericVector& self, py::array_t<dolfin::la_index> indices)
           {
             if (indices.ndim() > 1)
               throw py::index_error("Indices must be a 1D array");
             check_indices(indices, self.local_size());

             py::array_t<double> values(indices.size());
             self.get_local(values.mutable_data(), values.size(), indices.data());
             return values;
           })
      .def("__getitem__", [](dolfin::GenericVector& self, dolfin::la_index index)
           {
             if (self.local_size() == 0)
               throw py::index_error("GenericVector has zero (local) length. Cannot index into it.");
             else if (index < 0)
               throw py::index_error("Index is negative");
             else if (!(index < (dolfin::la_index) self.local_size()))
               throw py::index_error("Index exceeds (local) size of GenericVector");
             return self.getitem(index);
           })
      // __setitem__
      .def("__setitem__", [](dolfin::GenericVector& self, py::slice slice, double value)
           {
             std::size_t start, stop, step, slicelength;
             if (!slice.compute(self.size(), &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             if (start != 0 or stop != self.size() or step != 1)
               throw std::range_error("Only setting full slices for GenericVector is supported");

             self = value;
           })
      .def("__setitem__", [](dolfin::GenericVector& self, py::slice slice, const dolfin::GenericVector& x)
           {
             std::size_t start, stop, step, slicelength;
             if (!slice.compute(self.size(), &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             if (start != 0 or stop != self.size() or step != 1)
               throw std::range_error("Only setting full slices for GenericVector is supported");
             self = x;
           })
      .def("__setitem__", [](dolfin::GenericVector& self, py::slice slice, const py::array_t<double> x)
           {
             if (x.ndim() != 1)
               throw py::index_error("Values to set must be a 1D array");

             std::size_t start, stop, step, slicelength;
             if (!slice.compute(self.size(), &start, &stop, &step, &slicelength))
               throw py::error_already_set();
             if (start != 0 or stop != self.size() or step != 1)
               throw std::range_error("Only full slices are supported");

             std::vector<double> values(x.data(), x.data() + x.size());
             if (!values.empty())
             {
               self.set_local(values);
               self.apply("insert");
             }
           })
      .def("__setitem__", [](dolfin::GenericVector& self, const py::array_t<dolfin::la_index> indices, double x)
           {
             if (indices.ndim() > 1)
               throw py::index_error("Indices to set must be a 1D array");
             check_indices(indices, self.local_size());

             // FIXME: combine with  py::array_t<double> x version?
             std::vector<double> _x(indices.size(), x);
             self.set_local(_x.data(), _x.size(), indices.data());
             self.apply("insert");
           })
      .def("__setitem__", [](dolfin::GenericVector& self, const py::array_t<dolfin::la_index> indices,
                             const py::array_t<double> x)
           {
             if (indices.ndim() != 1)
               throw py::index_error("Indices to set must be a 1D array");
             if (x.ndim() != 1)
               throw py::index_error("Values to set must be a 1D array");
             if (indices.shape(0) != x.shape(0))
               throw py::index_error("Index mismatch");
             check_indices(indices, self.local_size());

             // FIXME: check sizes
             self.set_local(x.data(), x.size(), indices.data());
             self.apply("insert");
           })
      .def("__len__", [](dolfin::GenericVector& self) { return self.local_size(); })
      .def("size",  (std::size_t (dolfin::GenericVector::*)() const) &dolfin::GenericVector::size)
      //
      .def("get_local", [](const dolfin::GenericVector& instance,
                           const std::vector<dolfin::la_index>& rows)
           {
             py::array_t<double> data(rows.size());
             instance.get_local(data.mutable_data(), rows.size(), rows.data());
             return data;
           })
      .def("get_local", [](const dolfin::GenericVector& instance)
           {
             std::vector<double> values;
             instance.get_local(values);
             return py::array_t<double>(values.size(), values.data());
           })
      .def("set_local", [](dolfin::GenericVector& instance, std::vector<double> values)
           {
             std::vector<dolfin::la_index> indices(values.size());
             std::iota(indices.begin(), indices.end(), 0);
             instance.set_local(values.data(), values.size(), indices.data());
           })
      .def("add_local", [](dolfin::GenericVector& self, py::array_t<double> values)
           {
             assert(values.ndim() == 1);
             dolfin::Array<double> _values(values.size(), values.mutable_data());
             self.add_local(_values);
           })
      .def("gather", [](const dolfin::GenericVector& instance, dolfin::GenericVector& y,
                        const std::vector<dolfin::la_index>& rows)
           { instance.gather(y, rows); })
      .def("gather", [](const dolfin::GenericVector& instance, py::array_t<dolfin::la_index> rows)
           {
             std::vector<dolfin::la_index> _rows(rows.data(), rows.data() + rows.size());
             std::vector<double> values(rows.size());
             instance.gather(values, _rows);
             return py::array_t<double>(values.size(), values.data());
           })
      .def("gather", [](const dolfin::GenericVector& instance, std::vector<dolfin::la_index> rows)
           {
             std::vector<double> values(rows.size());
             instance.gather(values, rows);
             return py::array_t<double>(values.size(), values.data());
           })
      .def("gather_on_zero", [](const dolfin::GenericVector& instance)
           {
             std::vector<double> values;
             instance.gather_on_zero(values);
             return py::array_t<double>(values.size(), values.data());
           })
      .def("axpy", &dolfin::GenericVector::axpy)
      .def("sum", (double (dolfin::GenericVector::*)() const) &dolfin::GenericVector::sum)
      .def("sum", [](const dolfin::GenericVector& self, py::array_t<std::size_t> rows)
           { const dolfin::Array<std::size_t> _rows(rows.size(), rows.mutable_data()); return self.sum(_rows); })
      .def("max", (double (dolfin::GenericVector::*)() const) &dolfin::GenericVector::max)
      .def("min", (double (dolfin::GenericVector::*)() const) &dolfin::GenericVector::min)
      .def("inner", &dolfin::GenericVector::inner)
      .def("norm", &dolfin::GenericVector::norm)
      .def("local_size", &dolfin::GenericVector::local_size)
      .def("local_range", (std::pair<std::int64_t, std::int64_t> (dolfin::GenericVector::*)() const) &dolfin::GenericVector::local_range)
      .def("owns_index", &dolfin::GenericVector::owns_index)
      .def("apply", &dolfin::GenericVector::apply)
      .def("array", [](const dolfin::GenericVector& instance)
           {
             std::vector<double> values;
             instance.get_local(values);
             return py::array_t<double>(values.size(), values.data());
           })
      .def_property_readonly("__array_priority__", [](const dolfin::GenericVector& self){ return 0; });



    // dolfin::Matrix
    py::class_<dolfin::Matrix, std::shared_ptr<dolfin::Matrix>, dolfin::GenericMatrix>
      (m, "Matrix", "DOLFIN Matrix object")
      .def(py::init<>())
      .def(py::init<const dolfin::Matrix&>())  // Remove? (use copy instead)
      .def(py::init<const dolfin::GenericMatrix&>())  // Remove? (use copy instead)
      .def(py::init<MPI_Comm>()) // This comes last of constructors so pybind11 attempts it lasts (avoid OpenMPI comm casting problems)
      // Enabling the below messes up the operators because pybind11
      // then fails to try the GenericMatrix __mul__ operators
      /*
      .def("__mul__", [](const dolfin::Matrix& self, const dolfin::GenericVector& x)
           {
             // Specialised verion in place of GenericMatrix.__mul__
             // becuase this guarantees that a dolfin::Vector is
             // return rather than a more concrete vector. Maybe not
             // important, but some tests check the type.
             dolfin::Vector y(x.mpi_comm());
             self.init_vector(y, 0);
             self.mult(x, y);
             return y;
           }, py::is_operator(),py::arg().noconvert())
      */
      .def("instance", (std::shared_ptr<dolfin::LinearAlgebraObject>(dolfin::Matrix::*)())
           &dolfin::Matrix::shared_instance);

    // dolfin::Vector
    py::class_<dolfin::Vector, std::shared_ptr<dolfin::Vector>, dolfin::GenericVector>
      (m, "Vector", "DOLFIN Vector object")
      .def(py::init<>())
      .def(py::init<const dolfin::Vector&>())
      .def(py::init<const dolfin::GenericVector&>())
      .def(py::init<MPI_Comm>())
      .def(py::init<MPI_Comm, std::size_t>())
      .def("min", &dolfin::Vector::min)
      .def("max", &dolfin::Vector::max)
      .def("abs", &dolfin::Vector::abs)
      .def("norm", &dolfin::Vector::norm)
      .def("inner", &dolfin::Vector::inner)
      .def("axpy", &dolfin::Vector::axpy)
      .def("zero", &dolfin::Vector::zero)
      .def("apply", &dolfin::Vector::apply)
      .def("str", &dolfin::Vector::str)
      .def("instance", (std::shared_ptr<dolfin::LinearAlgebraObject>(dolfin::Vector::*)())
           &dolfin::Vector::shared_instance);

    // dolfin::BlockMatrix
    py::class_<dolfin::BlockMatrix, std::shared_ptr<dolfin::BlockMatrix>>
      (m, "BlockMatrix")
      .def(py::init<std::size_t, std::size_t>(), py::arg("m")=0, py::arg("n")=0)
      .def("__getitem__", [](dolfin::BlockMatrix& self, py::tuple index)
           {
             if (index.size() != 2)
               throw py::key_error("Wrong number of indices");
             std::size_t i = index[0].cast<std::size_t>();
             std::size_t j = index[1].cast<std::size_t>();
             return self.get_block(i, j);
           })
      .def("__setitem__", [](dolfin::BlockMatrix& self, py::tuple index,
                             std::shared_ptr<dolfin::GenericMatrix> m)
           {
             if (index.size() != 2)
               throw py::key_error("Wrong number of indices");
             std::size_t i = index[0].cast<std::size_t>();
             std::size_t j = index[1].cast<std::size_t>();
             self.set_block(i, j, m);
           })
      .def("mult", &dolfin::BlockMatrix::mult, py::arg("x"), py::arg("y"), py::arg("transposed")=false);

    // dolfin::BlockVector
    py::class_<dolfin::BlockVector, std::shared_ptr<dolfin::BlockVector>>
      (m, "BlockVector")
      .def(py::init<std::size_t>())
      .def("__getitem__", [](dolfin::BlockVector& self, std::size_t index)
           { return self.get_block(index); })
      .def("__setitem__", [](dolfin::BlockVector& self, std::size_t index,
                             std::shared_ptr<dolfin::GenericVector> v)
           { self.set_block(index, v); });

    // dolfin::Scalar
    py::class_<dolfin::Scalar, std::shared_ptr<dolfin::Scalar>, dolfin::GenericTensor>
      (m, "Scalar")
      .def(py::init<>())
      .def(py::init<MPI_Comm>())
      .def("add_local_value", &dolfin::Scalar::add_local_value)
      .def("apply", &dolfin::Scalar::apply)
      .def("mpi_comm", &dolfin::Scalar::mpi_comm)
      .def("get_scalar_value", &dolfin::Scalar::get_scalar_value);

    class PyLinearOperator : public dolfin::LinearOperator
    {
      // dolfin::LinearOperator trampoline class

      using dolfin::LinearOperator::LinearOperator;

      // pybdind11 has some issues when passing by reference (due to
      // the return value policy), so the below is non-standard.  See
      // https://github.com/pybind/pybind11/issues/250.

      std::size_t size(std::size_t dim) const
      {
        PYBIND11_OVERLOAD_PURE(std::size_t, dolfin::LinearOperator, size, );
      }

      void mult(const dolfin::GenericVector& x, dolfin::GenericVector& y) const
      {
        PYBIND11_OVERLOAD_INT(void, dolfin::LinearOperator, "mult", &x, &y);
        py::pybind11_fail("Tried to call pure virtual function dolfin::LinearOpertor::mult");
      }
    };

    // dolfin::LinearOperator
    py::class_<dolfin::LinearOperator, std::shared_ptr<dolfin::LinearOperator>,
               PyLinearOperator, dolfin::GenericLinearOperator>
      (m, "LinearOperator")
      //.def(py::init<>())
      .def(py::init<const dolfin::GenericVector&, const dolfin::GenericVector&>());

    // dolfin::GenericLinearAlgebraFactory
    py::class_<dolfin::GenericLinearAlgebraFactory, std::shared_ptr<dolfin::GenericLinearAlgebraFactory>>
      (m, "GenericLinearAlgebraFactory", "DOLFIN GenericLinearAlgebraFactory object");

    // dolfin::DefaultFactory
    py::class_<dolfin::DefaultFactory, std::shared_ptr<dolfin::DefaultFactory>>
      (m, "DefaultFactory", "DOLFIN DefaultFactory object")
      .def(py::init<>())
      .def_static("factory", &dolfin::DefaultFactory::factory)
      .def("create_matrix", &dolfin::DefaultFactory::create_matrix)
      .def("create_vector", &dolfin::DefaultFactory::create_vector);

    // dolfin::EigenFactory
    py::class_<dolfin::EigenFactory, std::shared_ptr<dolfin::EigenFactory>,
      dolfin::GenericLinearAlgebraFactory>
      (m, "EigenFactory", "DOLFIN EigenFactory object")
      .def("instance", &dolfin::EigenFactory::instance)
      .def("create_matrix", &dolfin::EigenFactory::create_matrix)
      .def("create_vector", &dolfin::EigenFactory::create_vector);

    // dolfin::EigenVector
    py::class_<dolfin::EigenVector, std::shared_ptr<dolfin::EigenVector>,
               dolfin::GenericVector>
      (m, "EigenVector", "DOLFIN EigenVector object")
      .def(py::init<>())
      .def(py::init<MPI_Comm>())
      .def(py::init<MPI_Comm, std::size_t>())
      //.def("array", (std::shared_ptr<Eigen::VectorXd> (dolfin::EigenVector::*)()) &dolfin::EigenVector::vec);
      .def("array", [](dolfin::EigenVector& self) -> Eigen::Ref<Eigen::VectorXd> { return *self.vec(); } );

    // dolfin::EigenMatrix
    py::class_<dolfin::EigenMatrix, std::shared_ptr<dolfin::EigenMatrix>,
               dolfin::GenericMatrix>
      (m, "EigenMatrix", "DOLFIN EigenMatrix object")
      .def(py::init<>())
      .def(py::init<std::size_t, std::size_t>())
      .def("sparray", (dolfin::EigenMatrix::eigen_matrix_type& (dolfin::EigenMatrix::*)()) &dolfin::EigenMatrix::mat,
           py::return_value_policy::reference_internal)
      .def("data_view", [](dolfin::EigenMatrix& instance)
           {
             auto _data = instance.data();
             std::size_t nnz = std::get<3>(_data);

             Eigen::Map<const Eigen::VectorXi> rows(std::get<0>(_data), instance.size(0) + 1);
             Eigen::Map<const Eigen::VectorXi> cols(std::get<1>(_data), nnz);
             Eigen::Map<const Eigen::VectorXd> values(std::get<2>(_data), nnz);

             return py::make_tuple(rows, cols, values);
           },
           py::return_value_policy::reference_internal, "Return CSR matrix data as NumPy arrays (shared data)")
      .def("data", [](dolfin::EigenMatrix& instance)
           {
             auto _data = instance.data();
             std::size_t nnz = std::get<3>(_data);

             Eigen::VectorXi rows = Eigen::Map<const Eigen::VectorXi>(std::get<0>(_data), instance.size(0) + 1);
             Eigen::VectorXi cols = Eigen::Map<const Eigen::VectorXi>(std::get<1>(_data), nnz);
             Eigen::VectorXd values  = Eigen::Map<const Eigen::VectorXd>(std::get<2>(_data), nnz);

             return py::make_tuple(rows, cols, values);
           },
           py::return_value_policy::copy, "Return copy of CSR matrix data as NumPy arrays");

    // dolfin::GenericLinearSolver
    py::class_<dolfin::GenericLinearSolver, std::shared_ptr<dolfin::GenericLinearSolver>,
               dolfin::Variable>
      (m, "GenericLinearSolver", "DOLFIN GenericLinearSolver object");

    #ifdef HAS_PETSC
    py::class_<dolfin::PETScOptions>(m, "PETScOptions")
      .def_static("set", (void (*)(std::string)) &dolfin::PETScOptions::set)
      .def_static("set", (void (*)(std::string, bool)) &dolfin::PETScOptions::set)
      .def_static("set", (void (*)(std::string, int)) &dolfin::PETScOptions::set)
      .def_static("set", (void (*)(std::string, double)) &dolfin::PETScOptions::set)
      .def_static("set", (void (*)(std::string, std::string)) &dolfin::PETScOptions::set)
      .def_static("clear", (void (*)(std::string)) &dolfin::PETScOptions::clear)
      .def_static("clear", (void (*)()) &dolfin::PETScOptions::clear);

    py::class_<dolfin::PETScObject, std::shared_ptr<dolfin::PETScObject>>(m, "PETScObject");

    // dolfin::PETScFactory
    py::class_<dolfin::PETScFactory, std::shared_ptr<dolfin::PETScFactory>,
      dolfin::GenericLinearAlgebraFactory>
      (m, "PETScFactory", "DOLFIN PETScFactory object")
      .def("instance", &dolfin::PETScFactory::instance)
      .def("create_matrix", &dolfin::PETScFactory::create_matrix)
      .def("create_vector", &dolfin::PETScFactory::create_vector);

    // dolfin::PETScVector
    py::class_<dolfin::PETScVector, std::shared_ptr<dolfin::PETScVector>,
               dolfin::GenericVector, dolfin::PETScObject>
      (m, "PETScVector", "DOLFIN PETScVector object")
      .def(py::init<>())
      .def(py::init<MPI_Comm>())
      .def(py::init<MPI_Comm, std::size_t>())
      .def("get_options_prefix", &dolfin::PETScVector::get_options_prefix)
      .def("set_options_prefix", &dolfin::PETScVector::set_options_prefix)
      .def("update_ghost_values", &dolfin::PETScVector::update_ghost_values);

    // dolfin::PETScBaseMatrix
    py::class_<dolfin::PETScBaseMatrix, std::shared_ptr<dolfin::PETScBaseMatrix>,
               dolfin::PETScObject, dolfin::Variable>(m, "PETScBaseMatrix");

    // dolfin::PETScMatrix
    py::class_<dolfin::PETScMatrix, std::shared_ptr<dolfin::PETScMatrix>,
               dolfin::GenericMatrix, dolfin::PETScBaseMatrix>
      (m, "PETScMatrix", "DOLFIN PETScMatrix object")
      .def(py::init<>())
      .def(py::init<MPI_Comm>())
      .def("get_options_prefix", &dolfin::PETScMatrix::get_options_prefix)
      .def("set_options_prefix", &dolfin::PETScMatrix::set_options_prefix)
      .def("set_nullspace", &dolfin::PETScMatrix::set_nullspace)
      .def("set_near_nullspace", &dolfin::PETScMatrix::set_near_nullspace);

    py::class_<dolfin::PETScPreconditioner, std::shared_ptr<dolfin::PETScPreconditioner>>
      (m, "PETScPreconditioner", "DOLFIN PETScPreconditioner object")
      .def(py::init<std::string>(), py::arg("type")="default")
      .def("preconditioners", &dolfin::PETScPreconditioner::preconditioners);

    #endif

    #ifdef HAS_TRILINOS

    // dolfin::TpetraFactory
    py::class_<dolfin::TpetraFactory, std::shared_ptr<dolfin::TpetraFactory>,
      dolfin::GenericLinearAlgebraFactory>
      (m, "TpetraFactory", "DOLFIN TpetraFactory object")
      .def("instance", &dolfin::TpetraFactory::instance)
      .def("create_matrix", &dolfin::TpetraFactory::create_matrix)
      .def("create_vector", &dolfin::TpetraFactory::create_vector);

    // dolfin::TpetraVector
    py::class_<dolfin::TpetraVector, std::shared_ptr<dolfin::TpetraVector>,
               dolfin::GenericVector>
      (m, "TpetraVector", "DOLFIN TpetraVector object")
      .def(py::init<MPI_Comm>(), py::arg("comm")=MPI_COMM_WORLD)
      .def(py::init<MPI_Comm, std::size_t>());

    // dolfin::TpetraMatrix
    py::class_<dolfin::TpetraMatrix, std::shared_ptr<dolfin::TpetraMatrix>,
               dolfin::GenericMatrix>
      (m, "TpetraMatrix", "DOLFIN TpetraMatrix object")
      .def(py::init<>());

    // dolfin::TrilinosPreconditioner
    py::class_<dolfin::TrilinosPreconditioner, std::shared_ptr<dolfin::TrilinosPreconditioner>>
      (m, "TrilinosPreconditioner", "DOLFIN TrilinosPreconditioner object");

    // dolfin::MueluPreconditioner
    py::class_<dolfin::MueluPreconditioner, std::shared_ptr<dolfin::MueluPreconditioner>,
               dolfin::TrilinosPreconditioner, dolfin::Variable>
      (m, "MueluPreconditioner", "Muelu Preconditioner")
      .def(py::init<>());

    // dolfin::BelosKrylovSolver
    py::class_<dolfin::BelosKrylovSolver, std::shared_ptr<dolfin::BelosKrylovSolver>,
              dolfin::GenericLinearSolver>
      (m, "BelosKrylovSolver", "Belos KrylovSolver")
      .def(py::init<std::string, std::shared_ptr<dolfin::TrilinosPreconditioner>>())
      .def("set_operator", &dolfin::BelosKrylovSolver::set_operator)
      .def("set_operators", &dolfin::BelosKrylovSolver::set_operators)
      .def("solve", (std::size_t (dolfin::BelosKrylovSolver::*)
                     (dolfin::GenericVector&, const dolfin::GenericVector&))
           &dolfin::BelosKrylovSolver::solve)
      .def("solve", (std::size_t (dolfin::BelosKrylovSolver::*)
                     (const dolfin::GenericLinearOperator&,
                      dolfin::GenericVector&, const dolfin::GenericVector&))
           &dolfin::BelosKrylovSolver::solve);
    #endif

    // dolfin::LUSolver
    py::class_<dolfin::LUSolver, std::shared_ptr<dolfin::LUSolver>,
      dolfin::GenericLinearSolver>
    (m, "LUSolver", "DOLFIN LUSolver object")
      .def(py::init<>())
      .def(py::init<std::shared_ptr<const dolfin::GenericLinearOperator>, std::string>(),
           py::arg("A"), py::arg("method")="default")
      .def(py::init<MPI_Comm, std::shared_ptr<const dolfin::GenericLinearOperator>,
         std::string>(),
           py::arg("comm"), py::arg("A"), py::arg("method") = "default")
      .def("set_operator", &dolfin::LUSolver::set_operator)
      .def("solve", (std::size_t (dolfin::LUSolver::*)(dolfin::GenericVector&,
                                                       const dolfin::GenericVector&))
           &dolfin::LUSolver::solve)
      .def("solve", (std::size_t (dolfin::LUSolver::*)(const dolfin::GenericLinearOperator&,
                                                       dolfin::GenericVector&,
                                                       const dolfin::GenericVector&))
           &dolfin::LUSolver::solve);

    #ifdef HAS_PETSC
    // dolfin::PETScLUSolver
    py::class_<dolfin::PETScLUSolver, std::shared_ptr<dolfin::PETScLUSolver>,
      dolfin::GenericLinearSolver>
      (m, "PETScLUSolver", "DOLFIN PETScLUSolver object")
      .def(py::init<MPI_Comm, std::string>(), py::arg("comm"), py::arg("method")="default")
      .def(py::init<std::string>(), py::arg("method")="default")
      .def(py::init<MPI_Comm, std::shared_ptr<const dolfin::PETScMatrix>, std::string>(),
           py::arg("comm"), py::arg("A"), py::arg("method")="default")
      .def(py::init<std::shared_ptr<const dolfin::PETScMatrix>, std::string>(),
           py::arg("A"), py::arg("method")="default")
      .def("get_options_prefix", &dolfin::PETScLUSolver::get_options_prefix)
      .def("set_options_prefix", &dolfin::PETScLUSolver::set_options_prefix)
      .def("solve", (std::size_t (dolfin::PETScLUSolver::*)(dolfin::GenericVector&, const dolfin::GenericVector&))
           &dolfin::PETScLUSolver::solve);
    #endif

    // dolfin::KrylovSolver
    py::class_<dolfin::KrylovSolver, std::shared_ptr<dolfin::KrylovSolver>,
               dolfin::GenericLinearSolver>
      (m, "KrylovSolver", "DOLFIN KrylovSolver object")
      .def(py::init<std::string, std::string>(), py::arg("method")="default",
           py::arg("preconditioner")="default")
      .def(py::init<std::shared_ptr<const dolfin::GenericLinearOperator>,
           std::string, std::string>(), py::arg("A"),
           py::arg("method")="default", py::arg("preconditioner")="default")
      .def(py::init<MPI_Comm, std::shared_ptr<const dolfin::GenericLinearOperator>,
           std::string, std::string>(), py::arg("comm"), py::arg("A"),
           py::arg("method")="default", py::arg("preconditioner")="default")
      .def("set_operator", &dolfin::KrylovSolver::set_operator)
      .def("set_operators", &dolfin::KrylovSolver::set_operators)
      .def("solve", (std::size_t (dolfin::KrylovSolver::*)(dolfin::GenericVector&,
                                                           const dolfin::GenericVector&))
           &dolfin::KrylovSolver::solve);

    #ifdef HAS_PETSC
    // dolfin::PETScKrylovSolver
    py::class_<dolfin::PETScKrylovSolver, std::shared_ptr<dolfin::PETScKrylovSolver>,
               dolfin::GenericLinearSolver>
      petsc_ks(m, "PETScKrylovSolver", "DOLFIN PETScKrylovSolver object");

    petsc_ks.def(py::init<>())
      .def(py::init<std::string>())
      .def(py::init<std::string, std::string>())
      .def(py::init<std::string, std::shared_ptr<dolfin::PETScPreconditioner>>())
      .def(py::init<KSP>())
      .def("get_options_prefix", &dolfin::PETScKrylovSolver::get_options_prefix)
      .def("set_options_prefix", &dolfin::PETScKrylovSolver::set_options_prefix)
      .def("get_norm_type", (dolfin::PETScKrylovSolver::norm_type (dolfin::PETScKrylovSolver::*)() const)
           &dolfin::PETScKrylovSolver::get_norm_type)
      .def("set_norm_type", &dolfin::PETScKrylovSolver::set_norm_type)
      .def("set_operator",  (void (dolfin::PETScKrylovSolver::*)(std::shared_ptr<const dolfin::GenericLinearOperator>))
           &dolfin::PETScKrylovSolver::set_operator)
      .def("set_operators", (void (dolfin::PETScKrylovSolver::*)(std::shared_ptr<const dolfin::GenericLinearOperator>,
                                                                 std::shared_ptr<const dolfin::GenericLinearOperator>))
           &dolfin::PETScKrylovSolver::set_operators)
      .def("solve", (std::size_t (dolfin::PETScKrylovSolver::*)(dolfin::GenericVector&, const dolfin::GenericVector&))
           &dolfin::PETScKrylovSolver::solve)
      .def("set_from_options", &dolfin::PETScKrylovSolver::set_from_options)
      .def("set_reuse_preconditioner", &dolfin::PETScKrylovSolver::set_reuse_preconditioner)
      .def("set_dm", &dolfin::PETScKrylovSolver::set_dm)
      .def("set_dm_active", &dolfin::PETScKrylovSolver::set_dm_active)
      .def("ksp", &dolfin::PETScKrylovSolver::ksp);

    py::enum_<dolfin::PETScKrylovSolver::norm_type>(petsc_ks, "norm_type")
      .value("none", dolfin::PETScKrylovSolver::norm_type::none)
      .value("default_norm", dolfin::PETScKrylovSolver::norm_type::default_norm)
      .value("preconditioned", dolfin::PETScKrylovSolver::norm_type::preconditioned)
      .value("unpreconditioned", dolfin::PETScKrylovSolver::norm_type::unpreconditioned)
      .value("natural", dolfin::PETScKrylovSolver::norm_type::natural);
    #endif

    #ifdef HAS_SLEPC
    // dolfin::SLEPcEigenSolver
    py::class_<dolfin::SLEPcEigenSolver, std::shared_ptr<dolfin::SLEPcEigenSolver>, dolfin::Variable>(m, "SLEPcEigenSolver")
      .def(py::init<std::shared_ptr<const dolfin::PETScMatrix>>())
      .def(py::init<std::shared_ptr<const dolfin::PETScMatrix>, std::shared_ptr<const dolfin::PETScMatrix>>())
      // FIXME: The below must come after the other
      // constructors. Check the MPI_Comm caster raises appropriate
      // exceptions for pybind11 to move onto next interface.
      .def(py::init<MPI_Comm>())
      .def("set_options_prefix", &dolfin::SLEPcEigenSolver::set_options_prefix)
      .def("set_from_options", &dolfin::SLEPcEigenSolver::set_from_options)
      .def("get_options_prefix", &dolfin::SLEPcEigenSolver::get_options_prefix)
      .def("get_number_converged", &dolfin::SLEPcEigenSolver::get_number_converged)
      .def("set_deflation_space", &dolfin::SLEPcEigenSolver::set_deflation_space)
      .def("set_initial_space", &dolfin::SLEPcEigenSolver::set_initial_space)
      .def("solve", (void (dolfin::SLEPcEigenSolver::*)())
           &dolfin::SLEPcEigenSolver::solve)
      .def("solve", (void (dolfin::SLEPcEigenSolver::*)(std::size_t))
           &dolfin::SLEPcEigenSolver::solve)
      .def("get_eigenvalue", [](dolfin::SLEPcEigenSolver& self, std::size_t i)
           {
             double lr, lc;
             self.get_eigenvalue(lr, lc, i);
             return py::make_tuple(lr, lc);
           })
      .def("get_eigenpair", [](dolfin::SLEPcEigenSolver& self, std::size_t i)
           {
             double lr, lc;
             dolfin::PETScVector r, c;
             self.get_eigenpair(lr, lc, r, c, i);
             return py::make_tuple(lr, lc, r, c);
           });
    #endif

    // dolfin::VectorSpaceBasis
    py::class_<dolfin::VectorSpaceBasis, std::shared_ptr<dolfin::VectorSpaceBasis>>(m, "VectorSpaceBasis")
      .def(py::init<const std::vector<std::shared_ptr<dolfin::GenericVector>>>())
      .def("is_orthonormal", &dolfin::VectorSpaceBasis::is_orthonormal, py::arg("tol")=1.0e-10)
      .def("is_orthogonal", &dolfin::VectorSpaceBasis::is_orthogonal, py::arg("tol")=1.0e-10)
      .def("orthogonalize", &dolfin::VectorSpaceBasis::orthogonalize)
      .def("orthonormalize", &dolfin::VectorSpaceBasis::orthonormalize, py::arg("tol")=1.0e-10)
      .def("dim", &dolfin::VectorSpaceBasis::dim)
      .def("__getitem__", &dolfin::VectorSpaceBasis::operator[]);

    // test_nullspace.h
    m.def("in_nullspace", &dolfin::in_nullspace, py::arg("A"), py::arg("x"), py::arg("type")="right");

    // la free functions
    m.def("has_linear_algebra_backend", &dolfin::has_linear_algebra_backend);
    m.def("linear_algebra_backends", &dolfin::linear_algebra_backends);
    m.def("has_krylov_solver_method", &dolfin::has_krylov_solver_method);
    m.def("has_krylov_solver_preconditioner", &dolfin::has_krylov_solver_preconditioner);
    m.def("normalize", &dolfin::normalize, py::arg("x"), py::arg("normalization_type")="average");

    // la/solve free functions
    m.def("solve", (std::size_t (*)(const dolfin::GenericLinearOperator&, dolfin::GenericVector&,
                                    const dolfin::GenericVector&, std::string, std::string)) &dolfin::solve,
          py::arg("A"), py::arg("x"), py::arg("b"), py::arg("method")="lu",
          py::arg("preconditioner")="none");
  }
}
