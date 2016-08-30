/* -*- C -*- */
// ===========================================================================
// SWIG directives for the DOLFIN la kernel module (post)
//
// The directives in this file are applied _after_ the header files of the
// modules has been loaded.
// ===========================================================================

// ---------------------------------------------------------------------------
// SLEPc specific extension code
// ---------------------------------------------------------------------------
#ifdef HAS_SLEPC
%feature("docstring") dolfin::SLEPcEigenSolver::_get_eigenvalue "Missing docstring";
%feature("docstring") dolfin::SLEPcEigenSolver::_get_eigenpair "Missing docstring";
%extend dolfin::SLEPcEigenSolver {

PyObject* _get_eigenvalue(const int i) {
    double lr, lc;
    self->get_eigenvalue(lr, lc, i);
    return Py_BuildValue("dd", lr, lc);
}

PyObject* _get_eigenpair(dolfin::PETScVector& r, dolfin::PETScVector& c, const int i) {
    double lr, lc;
    self->get_eigenpair(lr, lc, r, c, i);
    return Py_BuildValue("dd", lr, lc);
}

%pythoncode %{
    def get_eigenpair(self, i = 0, r_vec = None, c_vec = None,):
        """Gets the i-th solution of the eigenproblem"""
        r_vec = r_vec or PETScVector()
        c_vec = c_vec or PETScVector()
        lr, lc = self._get_eigenpair(r_vec, c_vec, i)
        return lr, lc, r_vec, c_vec

    def get_eigenvalue(self, i = 0):
        """Gets the i-th eigenvalue of the eigenproblem"""
        return self._get_eigenvalue(i)
%}

}
#endif

// ---------------------------------------------------------------------------
// C++ and Python extension code for BlockVector
// ---------------------------------------------------------------------------
%extend dolfin::BlockVector {
  %pythoncode %{

    def __getitem__(self, i):
        return self.get_block(i)

    def __setitem__(self, i, m):
        if not hasattr(self, "_items"):
            self._items = {}
        self._items[i] = m
        self.set_block(i, m)

    def __add__(self, v):
      a = self.copy()
      a += v
      return a

    def __sub__(self, v):
      a = self.copy()
      a -= v
      return a

    def __mul__(self, v):
      a = self.copy()
      a *= v
      return a

    def __rmul__(self, v):
      return self.__mul__(v)
  %}
}

// ---------------------------------------------------------------------------
// C++ and Python extension code for BlockMatrix
// ---------------------------------------------------------------------------
%extend dolfin::BlockMatrix {
%pythoncode
%{
    def __mul__(self, other):
        v = BlockVector(self.size(0))
        self.mult(other, v)
        return v

    def __getitem__(self, t):
        i,j = t
        return self.get_block(i, j)

    def __setitem__(self, t, m):
        if not hasattr(self, "_items"):
            self._items = {}
        self._items[t] = m
        i,j = t
        self.set_block(i, j, m)

%}
}

%pythoncode
%{
def la_index_dtype():
    "Return the numpy dtype equivalent to the type of la_index"
    from numpy import intc, int64
    return intc if common.sizeof_la_index() == 4 else int64
%}

// ---------------------------------------------------------------------------
// Modify the GenericVector interface
// ---------------------------------------------------------------------------
%feature("docstring") dolfin::GenericVector::_scale "Missing docstring";
%feature("docstring") dolfin::GenericVector::_vec_mul "Missing docstring";
%extend dolfin::GenericVector
{
  void _iadd_scalar(double a)
  {
    (*self) += a;
  }

  void _scale(double a)
  {
    (*self) *= a;
  }

  void _vec_mul(const GenericVector& other)
  {
    (*self) *= other;
  }
// ---------------------------------------------------------------------------
  %pythoncode
  %{
    def __in_parallel(self):
        first, last = self.local_range()
        return first > 0 or len(self) > last

    def __is_compatible(self, other):
        "Returns True if self, and other are compatible Vectors"
        if not isinstance(other, GenericVector):
            return False
        self_type = get_tensor_type(self)
        return self_type == get_tensor_type(other)

    def array(self):
        "Return a numpy array representation of the local part of a Vector"
        return self.get_local()

    def __contains__(self, value):
        from numpy import isscalar
        if isscalar(value):
            return value in self.get_local()
        return NotImplemented

    def __gt__(self, value):
        from numpy import isscalar
        if isscalar(value):
            return self.get_local() > value
        if isinstance(value, GenericVector):
            return self.get_local() > value.get_local()
        return NotImplemented

    def __ge__(self, value):
        from numpy import isscalar
        if isscalar(value):
            return self.get_local() >= value
        if isinstance(value, GenericVector):
            return self.get_local() >= value.get_local()
        return NotImplemented

    def __lt__(self, value):
        from numpy import isscalar
        if isscalar(value):
            return self.get_local() < value
        if isinstance(value, GenericVector):
            return self.get_local() < value.get_local()
        return NotImplemented

    def __le__(self, value):
        from numpy import isscalar
        if isscalar(value):
            return self.get_local() <= value
        if isinstance(value, GenericVector):
            return self.get_local() <= value.get_local()
        return NotImplemented

    def __eq__(self, value):
        from numpy import isscalar
        if isscalar(value):
            return self.get_local() == value
        if isinstance(value, GenericVector):
            return self.get_local() == value.get_local()
        return NotImplemented

    def __neq__(self, value):
        from numpy import isscalar
        if isscalar(value):
            return self.get_local() != value
        if isinstance(value, GenericVector):
            return self.get_local() != value.get_local()
        return NotImplemented

    def __neg__(self):
        ret = self.copy()
        ret *= -1
        return ret

    def __delitem__(self,i):
        raise ValueError("cannot delete Vector elements")

    def __delslice__(self,i,j):
        raise ValueError("cannot delete Vector elements")

    def __setslice__(self, i, j, values):
        from numpy import ndarray, arange
        if i == 0 and (j >= len(self) or j == -1): # slice == whole
            from numpy import isscalar
            # No test for equal lengths because this is checked by DOLFIN in _assign
            if isinstance(values, GenericVector) or isscalar(values):
                self._assign(values)
                return
            elif isinstance(values, ndarray):
                indices = arange(self.local_size(), dtype=la_index_dtype())
                self[indices] = values
                return

        raise IndexError("can only set full slices v[:]")

    def __getslice__(self, i, j):
        if i == 0 and (j >= len(self) or j == -1):
            return self.copy()
        raise IndexError("can only return full slices v[:]")

    def _check_indices(self, indices):
        from numpy import asarray, ndarray

        # Only accpect list and ndarrays
        if not isinstance(indices, (ndarray, list)):
            raise TypeError("expected an int or a list or numpy array of "\
                            "integers or a boolean numpy array as indices.")

        # If boolean array
        elif isinstance(indices, ndarray) and indices.dtype==bool:
            indices = indices.nonzero()[0]

        # Convert to correct indextypes
        if isinstance(indices, ndarray):

            # For some obscure reason we need to compare the char
            # attribute of the dtype to be able to differentiate
            # between correct dtypes. And to get the char attribute we
            # need to instantiate the bloody dtype and then access the
            # fraking dtype of the dtype, which can be asked for its
            # char...
            if indices.dtype.char != la_index_dtype()().dtype.char:
                indices = indices.astype(la_index_dtype())
        else:
            indices = asarray(indices, dtype=la_index_dtype())

        # Check range
        # FIXME: What should local_size mean?
        if not ((0<=indices).all() and (indices<self.local_size()).all()):
            raise IndexError("expected indices to be in [0..{}]".format(\
                self.local_size()))

        return indices

    def __getitem__(self, indices):
        """Return values corresponding to the given local indices"""
        from numpy import ndarray, integer, long, array, zeros, float_

        # If indices is a slice
        if isinstance(indices, slice):
            if not (indices.start is None and indices.stop is None and \
                    indices.step is None):
                raise IndexError("can only return full slices v[:]")
            return self.__getslice__(0, len(self))

        elif isinstance(indices, (int, integer, long)):
            indices = array([indices], dtype=la_index_dtype())

        indices = self._check_indices(indices)

        values = zeros(len(indices), dtype=float_)
        if len(values)>0:
            self.get_local(values, indices)
        return values

    def __setitem__(self, indices, values):
        """Set values corresponding to the given local indices

        This method is collective and user need to take care when this
        method is called, preventing MPI-deadlocks.
        """
        from numpy import asarray, ndarray, array, integer, isscalar, long, float_, ones
        try:

            # If indices is a slice
            if isinstance(indices, slice):
                if not (indices.start is None and indices.stop is None and \
                        indices.step is None):
                    raise IndexError("can only set full slices v[:]")
                self.__setslice__(0, len(self), values)
                return

            # If indices is a single integer
            elif isinstance(indices, (int, integer, long)):
                if isscalar(values):
                    indices = array([indices], dtype=la_index_dtype())
                else:
                    raise TypeError("provide a scalar to set single item")

            # Check indices
            indices = self._check_indices(indices)

            # Check passed values and convert scalar to ndarray of that value
            if isscalar(values):
                vec_values = ones(len(indices), dtype=float_)
                vec_values *= values
                values = vec_values

            elif isinstance(values, GenericVector):
                values = values.get_local()

            elif isinstance(values, ndarray):
                values = asarray(values, dtype=float_)

            else:
                raise TypeError("provide a scalar, GenericVector or numpy array of "\
                                "float to set items in Vector")

            if len(values) != len(indices):
                raise IndexError("expected same size of indices and values")

            # If values passed.
            if len(values) > 0:
                self.set_local(values, indices)

        finally:
            # Always call apply insert to avoid MPI dead locks if one or more
            # ranks fails
            self.apply("insert")

    def __len__(self):
        return self.size()

    def __iter__(self):
        for i in range(self.size()):
            yield self[i]

    def __add__(self, other):
        """x.__add__(y) <==> x+y"""
        from numpy import isscalar
        if isscalar(other):
            ret = self.copy()
            ret._iadd_scalar(other)
            return ret
        elif self.__is_compatible(other):
            ret = self.copy()
            ret.axpy(1.0, other)
            return ret
        return NotImplemented

    def __sub__(self, other):
        """x.__sub__(y) <==> x-y"""
        from numpy import isscalar
        if isscalar(other):
            ret = self.copy()
            ret._iadd_scalar(-other)
            return ret
        elif self.__is_compatible(other):
            ret = self.copy()
            ret.axpy(-1.0, other)
            return ret
        return NotImplemented

    def __mul__(self, other):
        """x.__mul__(y) <==> x*y"""
        from numpy import isscalar
        if isscalar(other):
            ret = self.copy()
            ret._scale(other)
            return ret
        if isinstance(other,GenericVector):
            ret = self.copy()
            ret._vec_mul(other)
            return ret
        return NotImplemented

    def __div__(self,other):
        """x.__div__(y) <==> x/y"""
        from numpy import isscalar
        if isscalar(other):
            ret = self.copy()
            ret._scale(1.0 / other)
            return ret
        return NotImplemented

    def __truediv__(self,other):
        """x.__div__(y) <==> x/y"""
        from numpy import isscalar
        if isscalar(other):
            ret = self.copy()
            ret._scale(1.0 / other)
            return ret
        return NotImplemented

    def __radd__(self,other):
        """x.__radd__(y) <==> y+x"""
        return self.__add__(other)

    def __rsub__(self, other):
        """x.__rsub__(y) <==> y-x"""
        ret = self.__sub__(other)
        ret.__imul__(-1.0)
        return ret

    def __rmul__(self, other):
        """x.__rmul__(y) <==> y*x"""
        from numpy import isscalar
        if isscalar(other):
            ret = self.copy()
            ret._scale(other)
            return ret
        return NotImplemented

    def __rdiv__(self, other):
        """x.__rdiv__(y) <==> y/x"""
        return NotImplemented

    def __iadd__(self, other):
        """x.__iadd__(y) <==> x+y"""
        from numpy import isscalar
        if isscalar(other):
            self._iadd_scalar(other)
            return self
        elif self.__is_compatible(other):
            self.axpy(1.0, other)
            return self
        return NotImplemented

    def __isub__(self, other):
        """x.__isub__(y) <==> x-y"""
        from numpy import isscalar
        if isscalar(other):
            self._iadd_scalar(-other)
            return self
        elif self.__is_compatible(other):
            self.axpy(-1.0, other)
            return self
        return NotImplemented

    def __imul__(self, other):
        """x.__imul__(y) <==> x*y"""
        from numpy import isscalar
        if isscalar(other):
            self._scale(other)
            return self
        if isinstance(other, GenericVector):
            self._vec_mul(other)
            return self
        return NotImplemented

    def __idiv__(self, other):
        """x.__idiv__(y) <==> x/y"""
        from numpy import isscalar
        if isscalar(other):
            self._scale(1.0 / other)
            return self
        return NotImplemented

    def __itruediv__(self, other):
        """x.__idiv__(y) <==> x/y"""
        from numpy import isscalar
        if isscalar(other):
            self._scale(1.0 / other)
            return self
        return NotImplemented

    def __iter__(self):
        return iter(self.array())
  %}
}

// ---------------------------------------------------------------------------
// Modify the EigenMatrix interface
// ---------------------------------------------------------------------------
%feature("docstring") dolfin::EigenMatrix::_data "Missing docstring";
%extend dolfin::EigenMatrix
{
  PyObject* _data()
  {
    PyObject* rows = %make_numpy_array(1, int)(self->size(0)+1,
                                               std::get<0>(self->data()),
                                               false);
    PyObject* cols = %make_numpy_array(1, int)(std::get<3>(self->data()),
                                               std::get<1>(self->data()),
                                               false);
    PyObject* values = %make_numpy_array(1, double)(std::get<3>(self->data()),
                                                    std::get<2>(self->data()),
                                                    false);

    if ( rows == NULL || cols == NULL || values == NULL)
      return NULL;

    return Py_BuildValue("NNN", rows, cols, values);
  }

  %pythoncode
  %{
    def data(self, deepcopy=True):
        """
        Return arrays to underlaying compresssed row/column storage data

        This method is only available for the Eigen linear algebra backend.

        *Arguments*
            deepcopy
                Return a copy of the data. If set to False a reference
                to the Matrix need to be kept, otherwise the data will be
                destroyed together with the destruction of the Matrix
        """
        rows, cols, values = self._data()
        if deepcopy:
            rows, cols, values = rows.astype(int), cols.astype(int), values.copy()
        else:
            _attach_base_to_numpy_array(rows, self)
            _attach_base_to_numpy_array(cols, self)
            _attach_base_to_numpy_array(values, self)

        return rows, cols, values

    def sparray(self):
        "Return a scipy.sparse representation of Matrix"
        from scipy.sparse import csr_matrix
        data = self.data(deepcopy=True)
        C = csr_matrix((data[2], data[1], data[0]))
        return C
  %}
}
// ---------------------------------------------------------------------------
// Modify the GenericMatrix interface
// ---------------------------------------------------------------------------
%feature("docstring") dolfin::GenericMatrix::_scale "Missing docstring";
%extend dolfin::GenericMatrix
{
  void _scale(double a)
  {
    (*self)*=a;
  }
  // -------------------------------------------------------------------------
  %pythoncode
  %{
    def __is_compatible(self,other):
        "Returns True if self, and other are compatible Vectors"
        if not isinstance(other,GenericMatrix):
            return False
        self_type = get_tensor_type(self)
        return self_type == get_tensor_type(other)

    def array(self):
        "Return a numpy array representation of Matrix"
        from numpy import zeros
        m_range = self.local_range(0);
        A = zeros((m_range[1] - m_range[0], self.size(1)))
        for i, row in enumerate(range(*m_range)):
            column, values = self.getrow(row)
            A[i, column] = values
        return A


    # FIXME: Getting matrix entries need to be carefully examined, especially
    #          for parallel objects.
    """
    def __getitem__(self,indices):
        from numpy import ndarray
        from types import SliceType
        if not (isinstance(indices, tuple) and len(indices) == 2):
            raise TypeError("expected two indices")
        if not all(isinstance(ind, (int, SliceType, list, ndarray)) for ind in indices):
            raise TypeError("an int, slice, list or numpy array as indices")

        if isinstance(indices[0], int):
            if isinstance(indices[1], int):
                return _get_matrix_single_item(self,indices[0],indices[1])
            return as_backend_type(_get_matrix_sub_vector(self,indices[0], indices[1], True))
        elif isinstance(indices[1],int):
            return as_backend_type(_get_matrix_sub_vector(self,indices[1], indices[0], False))
        else:
            same_indices = id(indices[0]) == id(indices[1])

            if not same_indices and ( type(indices[0]) == type(indices[1]) ):
                if isinstance(indices[0],(list,SliceType)):
                    same_indices = indices[0] == indices[1]
                else:
                    same_indices = (indices[0] == indices[1]).all()

            if same_indices:
                return as_backend_type(_get_matrix_sub_matrix(self, indices[0], None))
            else:
                return as_backend_type(_get_matrix_sub_matrix(self, indices[0], indices[1]))

    def __setitem__(self, indices, values):
        from numpy import ndarray, isscalar
        from types import SliceType
        if not (isinstance(indices, tuple) and len(indices) == 2):
            raise TypeError("expected two indices")
        if not all(isinstance(ind, (int, SliceType, list, ndarray)) for ind in indices):
            raise TypeError("an int, slice, list or numpy array as indices")

        if isinstance(indices[0], int):
            if isinstance(indices[1], int):
                if not isscalar(values):
                    raise TypeError("expected scalar for single value assigment")
                _set_matrix_single_item(self, indices[0], indices[1], values)
            else:
                raise NotImplementedError
                if isinstance(values,GenericVector):
                    _set_matrix_items_vector(self, indices[0], indices[1], values, True)
                elif isinstance(values,ndarray):
                    _set_matrix_items_array_of_float(self, indices[0], indices[1], values, True)
                else:
                    raise TypeError("expected a GenericVector or numpy array of float")
        elif isinstance(indices[1], int):
            raise NotImplementedError
            if isinstance(values, GenericVector):
                _set_matrix_items_vector(self, indices[1], indices[0], values, False)
            elif isinstance(values, ndarray):
                _set_matrix_items_array_of_float(self, indices[1], indices[0], values, False)
            else:
                raise TypeError("expected a GenericVector or numpy array of float")

        else:
            raise NotImplementedError
            same_indices = id(indices[0]) == id(indices[1])

            if not same_indices and ( type(indices[0]) == type(indices[1]) ):
                if isinstance(indices[0], (list, SliceType)):
                    same_indices = indices[0] == indices[1]
                else:
                    same_indices = (indices[0] == indices[1]).all()

            if same_indices:
                if isinstance(values,GenericMatrix):
                    _set_matrix_items_matrix(self, indices[0], None, values)
                elif isinstance(values, ndarray) and len(values.shape)==2:
                    _set_matrix_items_array_of_float(self, indices[0], None, values)
                else:
                    raise TypeError("expected a GenericMatrix or 2D numpy array of float")
            else:
                if isinstance(values,GenericMatrix):
                    _set_matrix_items_matrix(self, indices[0], indices[1], values)
                elif isinstance(values,ndarray) and len(values.shape) == 2:
                    _set_matrix_items_array_of_float(self, indices[0], indices[1], values)
                else:
                    raise TypeError("expected a GenericMatrix or 2D numpy array of float")
    """

    def __add__(self,other):
        """x.__add__(y) <==> x+y"""
        if self.__is_compatible(other):
            ret = self.copy()
            ret.axpy(1.0, other, False)
            return ret
        return NotImplemented

    def __sub__(self,other):
        """x.__sub__(y) <==> x-y"""
        if self.__is_compatible(other):
            ret = self.copy()
            ret.axpy(-1.0, other, False)
            return ret
        return NotImplemented

    def __mul__(self,other):
        """x.__mul__(y) <==> x*y"""
        from numpy import ndarray, isscalar
        if isscalar(other):
            ret = self.copy()
            ret._scale(other)
            return ret
        elif isinstance(other,GenericVector):
            matrix_type = get_tensor_type(self)
            vector_type = get_tensor_type(other)
            if vector_type not in _matrix_vector_mul_map[matrix_type]:
                raise TypeError("Provide a Vector which can be as_backend_typeed to ''"%vector_type.__name__)
            if type(other) == Vector:
                ret = Vector()
            else:
                ret = vector_type()
            self.mult(other, ret)
            return ret
        elif isinstance(other, ndarray):
            if len(other.shape) != 1:
                raise ValueError("Provide an 1D NumPy array")
            vec_size = other.shape[0]
            if vec_size != self.size(1):
                raise ValueError("Provide a NumPy array with length %d"%self.size(1))
            vec_type = _matrix_vector_mul_map[get_tensor_type(self)][0]
            vec = vec_type()
            vec.init(self.mpi_comm(), vec_size)
            vec.set_local(other)
            vec.apply("insert")
            result_vec = vec.copy()
            self.mult(vec, result_vec)
            #ret = other.copy()
            #result_vec.get_local(ret)
            #return ret
            return result_vec.get_local()
        return NotImplemented

    def __div__(self,other):
        """x.__div__(y) <==> x/y"""
        from numpy import isscalar
        if isscalar(other):
            ret = self.copy()
            ret._scale(1.0/other)
            return ret
        return NotImplemented

    def __truediv__(self,other):
        """x.__div__(y) <==> x/y"""
        from numpy import isscalar
        if isscalar(other):
            ret = self.copy()
            ret._scale(1.0/other)
            return ret
        return NotImplemented

    def __radd__(self,other):
        """x.__radd__(y) <==> y+x"""
        return self.__add__(other)

    def __rsub__(self,other):
        """x.__rsub__(y) <==> y-x"""
        return self.__sub__(other)

    def __rmul__(self,other):
        """x.__rmul__(y) <==> y*x"""
        from numpy import isscalar
        if isscalar(other):
            ret = self.copy()
            ret._scale(other)
            return ret
        return NotImplemented

    def __rdiv__(self,other):
        """x.__rdiv__(y) <==> y/x"""
        return NotImplemented

    def __iadd__(self,other):
        """x.__iadd__(y) <==> x+y"""
        if self.__is_compatible(other):
            self.axpy(1.0, other, False)
            return self
        return NotImplemented

    def __isub__(self,other):
        """x.__isub__(y) <==> x-y"""
        if self.__is_compatible(other):
            self.axpy(-1.0, other, False)
            return self
        return NotImplemented

    def __imul__(self,other):
        """x.__imul__(y) <==> x*y"""
        from numpy import isscalar
        if isscalar(other):
            self._scale(other)
            return self
        return NotImplemented

    def __idiv__(self,other):
        """x.__idiv__(y) <==> x/y"""
        from numpy import isscalar
        if isscalar(other):
            self._scale(1.0 / other)
            return self
        return NotImplemented

    def __itruediv__(self,other):
        """x.__idiv__(y) <==> x/y"""
        from numpy import isscalar
        if isscalar(other):
            self._scale(1.0 / other)
            return self
        return NotImplemented

  %}
}

// ---------------------------------------------------------------------------
// Macro with C++ and Python extension code for GenericVector types in PyDOLFIN
// that are able to return a pointer to the underlaying contigious data
// only used for the Eigen backend
// ---------------------------------------------------------------------------
%define LA_VEC_DATA_ACCESS(VEC_TYPE)
%feature("docstring") dolfin::VEC_TYPE::_data "Missing docstring";
%extend dolfin::VEC_TYPE
{
  PyObject* _data()
  {
    return %make_numpy_array(1, double)(self->size(), self->data(), false);
  }

  %pythoncode
  %{
    def data(self, deepcopy=True):
        """
        Return an array to underlaying data

        This method is only available for the Eigen linear algebra
        backend.

        *Arguments*
            deepcopy
                Return a copy of the data. If set to False a reference
                to the Matrix need to be kept, otherwise the data will be
                destroyed together with the destruction of the Matrix
        """
        ret = self._data()
        if deepcopy:
            ret = ret.copy()
        else:
            _attach_base_to_numpy_array(ret, self)

        return ret
  %}
}
%enddef

// ---------------------------------------------------------------------------
// Macro with code for down casting GenericTensors
// ---------------------------------------------------------------------------
%define AS_BACKEND_TYPE_MACRO(TENSOR_TYPE)
%inline %{
bool _has_type_ ## TENSOR_TYPE(const std::shared_ptr<dolfin::LinearAlgebraObject> tensor)
{ return dolfin::has_type<dolfin::TENSOR_TYPE>(*tensor); }

std::shared_ptr<dolfin::TENSOR_TYPE> _as_backend_type_ ## TENSOR_TYPE(const std::shared_ptr<dolfin::LinearAlgebraObject> tensor)
{ return dolfin::as_type<dolfin::TENSOR_TYPE>(tensor); }
%}

%pythoncode %{
_has_type_map[TENSOR_TYPE] = _has_type_ ## TENSOR_TYPE
_as_backend_type_map[TENSOR_TYPE] = _as_backend_type_ ## TENSOR_TYPE
%}

%enddef

// ---------------------------------------------------------------------------
// Run the data macro
// ---------------------------------------------------------------------------
LA_VEC_DATA_ACCESS(EigenVector)

// ---------------------------------------------------------------------------
// Define Python lookup maps for as_backend_typeing
// ---------------------------------------------------------------------------
%pythoncode %{
_has_type_map = {}
_as_backend_type_map = {}
# A map with matrix types as keys and list of possible vector types as values
_matrix_vector_mul_map = {}
%}

// ---------------------------------------------------------------------------
// Run the downcast macro
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Run backend specific macros
// ---------------------------------------------------------------------------
AS_BACKEND_TYPE_MACRO(EigenVector)
AS_BACKEND_TYPE_MACRO(EigenMatrix)
//AS_BACKEND_TYPE_MACRO(EigenLinearOperator)
%pythoncode %{
_matrix_vector_mul_map[EigenMatrix] = [EigenVector]
%}

#ifdef HAS_TRILINOS
AS_BACKEND_TYPE_MACRO(TpetraVector)
AS_BACKEND_TYPE_MACRO(TpetraMatrix)
%pythoncode %{
_matrix_vector_mul_map[TpetraMatrix] = [TpetraVector]
%}
#endif

#ifdef HAS_PETSC
AS_BACKEND_TYPE_MACRO(PETScVector)
AS_BACKEND_TYPE_MACRO(PETScMatrix)
AS_BACKEND_TYPE_MACRO(PETScLinearOperator)

%pythoncode %{
_matrix_vector_mul_map[PETScMatrix] = [PETScVector]
_matrix_vector_mul_map[PETScLinearOperator] = [PETScVector]
%}

#ifdef HAS_PETSC4PY
// Override default .mat() and .vec() calls.
// These are wrapped up by petsc4py typemaps so that
// we see a petsc4py object on the python side.

%feature("docstring") dolfin::PETScBaseMatrix::mat "Return petsc4py representation of PETSc Mat";
%extend dolfin::PETScBaseMatrix
{
  void mat(Mat &A)
  { A = self->mat(); }
}

%feature("docstring") dolfin::PETScVector::vec "Return petsc4py representation of PETSc Vec";
%extend dolfin::PETScVector
{
  void vec(Vec&v)
  { v = self->vec(); }
}

%feature("docstring") dolfin::PETScKrylovSolver::ksp "Return petsc4py representation of PETSc KSP solver";
%extend dolfin::PETScKrylovSolver
{
  void ksp(KSP& ksp)
  { ksp = self->ksp(); }
}

%feature("docstring") dolfin::PETScLUSolver::ksp "Return petsc4py representation of PETSc LU solver";
%extend dolfin::PETScLUSolver
{
  void ksp(KSP& ksp)
  { ksp = self->ksp(); }
}

%feature("docstring") dolfin::PETScSNESSolver::snes "Return petsc4py representation of PETSc SNES solver";
%extend dolfin::PETScSNESSolver
{
  void snes(SNES& snes)
  { snes = self->snes(); }
}

%feature("docstring") dolfin::PETScTAOSolver::tao "Return petsc4py representation of PETSc TAO solver";
%extend dolfin::PETScTAOSolver
{
  void tao(Tao& tao)
  { tao = self->tao(); }
}

#else
%extend dolfin::PETScBaseMatrix {
    %pythoncode %{
        def mat(self):
            common.dolfin_error("dolfin/swig/la/post.i",
                                "access PETScMatrix objects in python",
                                "dolfin must be configured with petsc4py enabled")
            return None
    %}
}

%extend dolfin::PETScVector {
    %pythoncode %{
        def vec(self):
            common.dolfin_error("dolfin/swig/la/post.i",
                                "access PETScVector objects in python",
                                "dolfin must be configured with petsc4py enabled")
            return None
    %}
}

%extend dolfin::PETScKrylovSolver {
    %pythoncode %{
        def ksp(self):
            common.dolfin_error("dolfin/swig/la/post.i",
                                "access PETScKrylovSolver objects in python",
                                "dolfin must be configured with petsc4py enabled")
            return None
    %}
}

%extend dolfin::PETScLUSolver {
    %pythoncode %{
        def ksp(self):
            common.dolfin_error("dolfin/swig/la/post.i",
                                "access PETScLUSolver objects in python",
                                "dolfin must be configured with petsc4py enabled")
            return None
    %}
}

%extend dolfin::PETScSNESSolver {
    %pythoncode %{
        def snes(self):
            common.dolfin_error("dolfin/swig/la/post.i",
                                "access PETScSNESSolver objects in python",
                                "dolfin must be configured with petsc4py enabled")
            return None
    %}
}

%extend dolfin::PETScTAOSolver {
    %pythoncode %{
        def tao(self):
            common.dolfin_error("dolfin/swig/la/post.i",
                                "access PETScTAOSolver objects in python",
                                "dolfin must be configured with petsc4py enabled")
            return None
    %}
}

#endif  // HAS_PETSC4PY
#endif  // HAS_PETSC

#ifdef HAS_SLEPC
#ifdef HAS_SLEPC4PY
// Override default SLEPcEigenSolver.eps() call.
// These are wrapped up by slepcc4py typemaps so that
// we see a slepc4py object on the python side.
%feature("docstring") dolfin::SLEPcEigenSolver::eps "Return slepc4py representation of SLEPc EPS";
%extend dolfin::SLEPcEigenSolver
{
  void eps(EPS &e)
  { e = self->eps(); }
}
#else
%extend dolfin::SLEPcEigenSolver{
    %pythoncode %{
        def eps(self):
            common.dolfin_error("dolfin/swig/la/post.i",
                                "access SLEPcEigenSolver objects in python",
                                "dolfin must be configured with slepc4py enabled")
            return None
    %}
}
#endif  // HAS_PETSC4PY
#endif  // HAS_PETSC

// ---------------------------------------------------------------------------
// Dynamic wrappers for GenericTensor::as_backend_type and
// GenericTensor::has_type, using dict of tensor types to select from
// C++ template instantiations
// ---------------------------------------------------------------------------
%pythoncode %{
def get_tensor_type(tensor):
    "Return the concrete subclass of tensor."
    for k, v in _has_type_map.items():
        if v(tensor):
            return k
    common.dolfin_error("dolfin/swig/la/post.i",
                        "extract backend type for %s" % type(tensor).__name__,
                        "Unable to get tensor type in Python layer")

def has_type(tensor, subclass):
    "Return wether tensor is of the given subclass."
    global _has_type_map
    assert _has_type_map
    assert subclass in _has_type_map
    return bool(_has_type_map[subclass](tensor))

def as_backend_type(tensor, subclass=None):
    "Cast tensor to the given subclass, passing the wrong class is an error."
    global _as_backend_type_map
    assert _as_backend_type_map
    if subclass is None:
        subclass = get_tensor_type(tensor)
    assert subclass in _as_backend_type_map
    ret = _as_backend_type_map[subclass](tensor)

    # Store the tensor to avoid garbage collection
    ret._org_upcasted_tensor = tensor
    return ret

%}
