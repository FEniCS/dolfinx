// Instantiate uBLAS matrix and Factory types
%template(uBLASSparseMatrix) dolfin::uBLASMatrix<dolfin::ublas_sparse_matrix>;
%template(uBLASDenseMatrix) dolfin::uBLASMatrix<dolfin::ublas_dense_matrix>;
%template(uBLASSparseFactory) dolfin::uBLASFactory<dolfin::ublas_sparse_matrix>;
%template(uBLASDenseFactory) dolfin::uBLASFactory<dolfin::ublas_dense_matrix>;

// Define names for uBLAS matrix types
// These are needed so returned type from down_cast get correctly wrapped
%typedef dolfin::uBLASMatrix<dolfin::ublas_sparse_matrix> uBLASSparseMatrix;
%typedef dolfin::uBLASMatrix<dolfin::ublas_dense_matrix>  uBLASDenseMatrix;

#ifdef HAS_SLEPC
%extend dolfin::SLEPcEigenSolver {

PyObject* getEigenvalue(const int emode) {
    double err, ecc;
    self->getEigenvalue(err, ecc, emode);

    PyObject* result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(err));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(ecc));
    Py_INCREF(result);
    return result;
}

PyObject* getEigenpair(dolfin::PETScVector& rr, dolfin::PETScVector& cc, const int emode) {
    double err, ecc;
    self->getEigenpair(err, ecc, rr, cc, emode);

    PyObject* result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(err));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(ecc));
    Py_INCREF(result);
    return result;
}

}
#endif

%extend dolfin::BlockVector {
    Vector& getitem(int i) 
    { 
      return self->get(i);
    }
    void setitem(int i, Vector& v)
    {
      self->set(i,v); 
    }
}

%extend dolfin::BlockVector {
  %pythoncode %{

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



%extend dolfin::BlockMatrix {
%pythoncode
%{
    def __mul__(self, other):
      v = BlockVector(self.size(0))
      self.mult(other, v)
      return v
%}
}

%pythoncode
%{
  def BlockMatrix_get(self,t):
    i,j = t
    return self.get(i,j)

  def BlockMatrix_set(self,t,m): 
    i,j = t 
    return self.set(i,j,m)

  BlockMatrix.__getitem__ = BlockMatrix_get
  BlockMatrix.__setitem__ = BlockMatrix_set
%}

// Vector la interface macro
%define LA_POST_VEC_INTERFACE(VEC_TYPE)
%extend dolfin::VEC_TYPE
{
  void _scale(double a)
  {
    (*self)*=a;
  }

  %pythoncode
  %{
    def __is_compatibable(self,other):
        "Returns True if self, and other are compatible Vectors"
        if not isinstance(other,GenericVector):
            return False
        self_type = get_tensor_type(self)
        return self_type == get_tensor_type(other)
        
    def array(self):
        " Return a numpy array representation of Vector"
        import numpy
        v = numpy.zeros(self.size())
        self.get(v)
        return v

    def __add__(self,other):
        """x.__add__(y) <==> x+y"""
        if self.__is_compatibable(other):
            ret = self.copy()
            ret.axpy(1.0,other)
            return ret
        return NotImplemented
    
    def __sub__(self,other):
        """x.__sub__(y) <==> x-y"""
        if self.__is_compatibable(other):
            ret = self.copy()
            ret.axpy(-1.0,other)
            return ret
        return NotImplemented
    
    def __mul__(self,other):
        """x.__mul__(y) <==> x*y"""
        if isinstance(other,(int,float)):
            ret = self.copy()
            ret._scale(other)
            return ret
        return NotImplemented
    
    def __div__(self,other):
        """x.__div__(y) <==> x/y"""
        if isinstance(other,(int,float)):
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
        if isinstance(other,(int,float)):
            ret = self.copy()
            ret._scale(other)
            return ret
        return NotImplemented
    
    def __rdiv__(self,other):
        """x.__rdiv__(y) <==> y/x"""
        return NotImplemented
    
    def __iadd__(self,other):
        """x.__iadd__(y) <==> x+y"""
        if self.__is_compatibable(other):
            self.axpy(1.0,other)
            return self
        return NotImplemented
    
    def __isub__(self,other):
        """x.__isub__(y) <==> x-y"""
        if self.__is_compatibable(other):
            self.axpy(-1.0,other)
            return self
        return NotImplemented
    
    def __imul__(self,other):
        """x.__imul__(y) <==> x*y"""
        if isinstance(other,(float,int)):
            self._scale(other)
            return self
        return NotImplemented

    def __idiv__(self,other):
        """x.__idiv__(y) <==> x/y"""
        if isinstance(other,(float,int)):
            self._scale(1.0/other)
            return self
        return NotImplemented
    
  %}
}
%enddef

// Matrix la interface macro
%define LA_POST_MAT_INTERFACE(MAT_TYPE)
%extend dolfin::MAT_TYPE
{
  void _scale(double a)
  {
    (*self)*=a;
  }
  
  PyObject* data() {
    npy_intp rowdims[1];
    rowdims[0] = self->size(0)+1;
    
    PyArrayObject* rows = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, rowdims, NPY_ULONG, (char *)(std::tr1::get<0>(self->data()))));
    if ( rows == NULL ) return NULL;
    PyArray_INCREF(rows);
    
    npy_intp coldims[1];
    coldims[0] = std::tr1::get<3>(self->data());
    
    PyArrayObject* cols = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, coldims, NPY_ULONG, (char *)(std::tr1::get<1>(self->data()))));
    if ( cols == NULL ) return NULL;
    PyArray_INCREF(cols);
    
    npy_intp valuedims[1];
    valuedims[0] = std::tr1::get<3>(self->data());
    
    PyArrayObject* values = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, valuedims, NPY_DOUBLE, (char *)(std::tr1::get<2>(self->data()))));
    if ( values == NULL ) return NULL;
    PyArray_INCREF(values);
    
    return PyTuple_Pack(3,rows, cols, values);
  }

  %pythoncode
  %{
    def __is_compatibable(self,other):
        "Returns True if self, and other are compatible Vectors"
        if not isinstance(other,GenericMatrix):
            return False
        self_type = get_tensor_type(self)
        return self_type == get_tensor_type(other)
        
    def array(self):
        " Return a numpy array representation of Matrix"
        import numpy
        A = numpy.zeros((self.size(0), self.size(1)))
        c = STLVectorUInt()
        v = STLVectorDouble()
        for i in xrange(self.size(0)):
            self.getrow(i, c, v)
            A[i,c] = v
        return A

    def __add__(self,other):
        """x.__add__(y) <==> x+y"""
        if self.__is_compatibable(other):
            ret = self.copy()
            ret.axpy(1.0,other)
            return ret
        return NotImplemented
    
    def __sub__(self,other):
        """x.__sub__(y) <==> x-y"""
        if self.__is_compatibable(other):
            ret = self.copy()
            ret.axpy(-1.0,other)
            return ret
        return NotImplemented
    
    def __mul__(self,other):
        """x.__mul__(y) <==> x*y"""
        from numpy import ndarray
        if isinstance(other,(int,float)):
            ret = self.copy()
            ret._scale(other)
            return ret
        elif isinstance(other,GenericVector):
            matrix_type = get_tensor_type(self)
            vector_type = get_tensor_type(other)
            if vector_type not in _matrix_vector_mul_map[matrix_type]:
                raise TypeError, "Provide a Vector which can be down_casted to ''"%vector_type.__name__
            if type(other) == Vector:
                ret = Vector(self.size(0))
            else:
                ret = vector_type(self.size(0))
            self.mult(other, ret)
            return ret
        elif isinstance(other,ndarray):
            if len(other.shape) !=1:
                raise ValueError, "Provide an 1D NumPy array"
            vec_size = other.shape[0]
            if vec_size != self.size(1):
                raise ValueError, "Provide a NumPy array with length %d"%self.size(1)
            vec_type = _matrix_vector_mul_map[get_tensor_type(self)][0]
            vec  = vec_type(vec_size)
            vec.set(other)
            result_vec = vec.copy()
            self.mult(vec, result_vec)
            ret = other.copy()
            result_vec.get(ret)
            return ret
        return NotImplemented
    
    def __div__(self,other):
        """x.__div__(y) <==> x/y"""
        if isinstance(other,(int,float)):
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
        if isinstance(other,(int,float)):
            ret = self.copy()
            ret._scale(other)
            return ret
        return NotImplemented
    
    def __rdiv__(self,other):
        """x.__rdiv__(y) <==> y/x"""
        return NotImplemented
    
    def __iadd__(self,other):
        """x.__iadd__(y) <==> x+y"""
        if self.__is_compatibable(other):
            self.axpy(1.0,other)
            return self
        return NotImplemented
    
    def __isub__(self,other):
        """x.__isub__(y) <==> x-y"""
        if self.__is_compatibable(other):
            self.axpy(-1.0,other)
            return self
        return NotImplemented
    
    def __imul__(self,other):
        """x.__imul__(y) <==> x*y"""
        if isinstance(other,(float,int)):
            self._scale(other)
            return self
        return NotImplemented

    def __idiv__(self,other):
        """x.__idiv__(y) <==> x/y"""
        if isinstance(other,(float,int)):
            self._scale(1.0/other)
            return self
        return NotImplemented

  %}
}
%enddef


// Vector la interface macro
%define LA_VEC_DATA_ACCESS(VEC_TYPE)
%extend dolfin::VEC_TYPE
{
  PyObject* _data()
  {
    npy_intp valuedims[1];
    valuedims[0] = self->size();
    PyArrayObject* values = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, valuedims, NPY_DOUBLE, (char *)(self->data())));
    if ( values == NULL ) return NULL;
    PyArray_INCREF(values);
    return reinterpret_cast<PyObject*>(values);
  }

  %pythoncode
  %{
    def data(self):
        " Return an array to the underlaying data"
        return self._data()
  %}
}
%enddef

// Down cast macro
%define DOWN_CAST_MACRO(TENSOR_TYPE)
%inline %{
bool has_type_ ## TENSOR_TYPE(dolfin::GenericTensor & tensor)
{ return tensor.has_type<dolfin::TENSOR_TYPE>(); }

dolfin::TENSOR_TYPE & down_cast_ ## TENSOR_TYPE(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::TENSOR_TYPE>(); }
%}

%pythoncode %{
_has_type_map[TENSOR_TYPE] = has_type_ ## TENSOR_TYPE
_down_cast_map[TENSOR_TYPE] = down_cast_ ## TENSOR_TYPE
%}

%enddef

// Initialize tensor type maps
%pythoncode %{
_has_type_map = {}
_down_cast_map = {}
# A map with matrix types as keys and list of possible vector types as values
_matrix_vector_mul_map = {}
%}

// Run the interface macros
LA_POST_VEC_INTERFACE(GenericVector)
LA_POST_MAT_INTERFACE(GenericMatrix)
//LA_VEC_DATA_ACCESS(GenericVector)

LA_POST_VEC_INTERFACE(Vector)
LA_POST_MAT_INTERFACE(Matrix)
LA_VEC_DATA_ACCESS(Vector)

LA_POST_VEC_INTERFACE(uBLASVector)
// NOTE: The uBLAS macros need to be run using the whole template type
// I have tried using the typmaped one from above but with no luck.
LA_POST_MAT_INTERFACE(uBLASMatrix<dolfin::ublas_sparse_matrix>)
LA_POST_MAT_INTERFACE(uBLASMatrix<dolfin::ublas_dense_matrix>)
LA_VEC_DATA_ACCESS(uBLASVector)

// Run the downcast macro
DOWN_CAST_MACRO(uBLASVector)
DOWN_CAST_MACRO(uBLASSparseMatrix)
DOWN_CAST_MACRO(uBLASDenseMatrix)

%pythoncode %{
_matrix_vector_mul_map[uBLASSparseMatrix] = [uBLASVector]
_matrix_vector_mul_map[uBLASDenseMatrix]  = [uBLASVector]
%}

#ifdef HAS_PETSC
LA_POST_VEC_INTERFACE(PETScVector)
LA_POST_MAT_INTERFACE(PETScMatrix)

DOWN_CAST_MACRO(PETScVector)
DOWN_CAST_MACRO(PETScMatrix)

%pythoncode %{
_matrix_vector_mul_map[PETScMatrix] = [PETScVector]
%}
#endif

#ifdef HAS_TRILINOS
LA_POST_VEC_INTERFACE(EpetraVector)
LA_POST_MAT_INTERFACE(EpetraMatrix)

DOWN_CAST_MACRO(EpetraVector)
DOWN_CAST_MACRO(EpetraMatrix)

%pythoncode %{
_matrix_vector_mul_map[EpetraMatrix] = [EpetraVector]
%}

%extend dolfin::EpetraMatrix
{
  Epetra_FECrsMatrix& ref_mat() const
  {
    return *self->mat();
  }
}

%extend dolfin::EpetraVector
{
  Epetra_FEVector& ref_vec() const
  {
    return *self->vec();
  }
}
#endif

#ifdef HAS_MTL4
LA_POST_VEC_INTERFACE(MTL4Vector)
LA_POST_MAT_INTERFACE(MTL4Matrix)
LA_VEC_DATA_ACCESS(MTL4Vector)

DOWN_CAST_MACRO(MTL4Vector)
DOWN_CAST_MACRO(MTL4Matrix)

%pythoncode %{
_matrix_vector_mul_map[MTL4Matrix] = [MTL4Vector]
%}
#endif

// Dynamic wrappers for GenericTensor::down_cast and GenericTensor::has_type, using dict of tensor types to select from C++ template instantiations
%pythoncode %{
def get_tensor_type(tensor):
    "Return the concrete subclass of tensor."
    for k,v in _has_type_map.items():
        if v(tensor):
            return k
    dolfin_error("Unregistered tensor type.")

def has_type(tensor, subclass):
    "Return wether tensor is of the given subclass."
    global _has_type_map
    assert _has_type_map
    assert subclass in _has_type_map
    return bool(_has_type_map[subclass](tensor))

def down_cast(tensor, subclass=None):
    "Cast tensor to the given subclass, passing the wrong class is an error."
    global _down_cast_map
    assert _down_cast_map
    if subclass is None:
        subclass = get_tensor_type(tensor)
    assert subclass in _down_cast_map
    ret = _down_cast_map[subclass](tensor)

    # Store the tensor to avoid garbage collection
    ret._org_upcasted_tensor = tensor
    return ret

%}

%feature("docstring") has_linear_algebra_backend "
Returns True if a linear algebra backend is available.
";

%inline %{
bool has_linear_algebra_backend(std::string backend)
{
  if (backend == "uBLAS")
  {
    return true;
  }
  else if (backend == "PETSc")
  {
#ifdef HAS_PETSC
    return true;
#else 
    return false;
#endif
  }
  else if (backend == "Epetra")
  {
#ifdef HAS_TRILINOS
    return true;
#else 
    return false;
#endif
  }
  else if (backend == "MTL4")
  {
#ifdef HAS_MTL4
    return true;
#else 
    return false;
#endif
  }
  else if (backend == "STL")
  {
    return true;
  }
  return false;
}
%}

