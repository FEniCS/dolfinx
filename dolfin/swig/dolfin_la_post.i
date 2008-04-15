// Instantiate uBlas matrix types
%template(uBlasSparseMatrix) dolfin::uBlasMatrix<dolfin::ublas_sparse_matrix>;
%template(uBlasDenseMatrix) dolfin::uBlasMatrix<dolfin::ublas_dense_matrix>;
// Define names for uBlas matrix types
%typedef dolfin::uBlasMatrix<dolfin::ublas_sparse_matrix> uBlasSparseMatrix;
%typedef dolfin::uBlasMatrix<dolfin::ublas_dense_matrix> uBlasDenseMatrix;


#ifdef HAS_SLEPC
%extend dolfin::SLEPcEigenvalueSolver{

PyObject* getEigenpair(dolfin::PETScVector& rr, dolfin::PETScVector& cc, const int emode) {
    dolfin::real err, ecc;
    self->getEigenpair(err, ecc, rr, cc, emode);

    PyObject* result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(err));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(ecc));
    Py_INCREF(result);
    return result;

}

}
#endif

%extend dolfin::Matrix {
  %pythoncode %{
    def __mul__(self, other):
      v = Vector(self.size(0))
      self.mult(other, v)
      return v

  %}
}

#ifdef HAS_PETSC
%extend dolfin::PETScMatrix {
  %pythoncode %{
    def __mul__(self, other):
      v = PETScVector(self.size(0))
      self.mult(other, v)
      return v

  %}
}
#else
%extend dolfin::uBlasMatrix {
  %pythoncode %{
    def __mul__(self, other):
      v = PETScVector(self.size(0))
      self.mult(other, v)
      return v

  %}
}
#endif


%extend dolfin::Vector {
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



#ifdef HAS_PETSC
%extend dolfin::PETScVector {
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
#else
%extend dolfin::uBlasVector {
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
#endif



// Initialize tensor type map
%pythoncode %{
_has_type_map = {}
_down_cast_map = {}
%}


// Add PETSc support
#ifdef HAS_PETSC

%inline %{

bool has_type_petsc_vector(dolfin::GenericTensor & tensor)
{ return tensor.has_type<dolfin::PETScVector>(); }

bool has_type_petsc_matrix(dolfin::GenericTensor & tensor)
{ return tensor.has_type<dolfin::PETScMatrix>(); }

dolfin::PETScVector & down_cast_petsc_vector(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::PETScVector>(); }

dolfin::PETScMatrix & down_cast_petsc_matrix(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::PETScMatrix>(); }
%}

%pythoncode %{
_has_type_map[PETScVector] = has_type_petsc_vector
_has_type_map[PETScMatrix] = has_type_petsc_matrix
_down_cast_map[PETScVector] = down_cast_petsc_vector
_down_cast_map[PETScMatrix] = down_cast_petsc_matrix
%}

#endif

// Add uBlas support
//#ifdef HAS_BOOST

%inline %{
bool has_type_ublas_vector(dolfin::GenericTensor & tensor)
{ return tensor.has_type<dolfin::uBlasVector>(); }

bool has_type_ublas_matrix(dolfin::GenericTensor & tensor)
{ return tensor.has_type<uBlasSparseMatrix>(); }

dolfin::uBlasVector & down_cast_ublas_vector(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::uBlasVector>(); }

uBlasSparseMatrix   & down_cast_ublas_matrix(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<uBlasSparseMatrix>(); }
%}

%pythoncode %{
_has_type_map[uBlasVector] = has_type_ublas_vector
_has_type_map[uBlasSparseMatrix] = has_type_ublas_matrix
_down_cast_map[uBlasVector] = down_cast_ublas_vector
_down_cast_map[uBlasSparseMatrix] = down_cast_ublas_matrix
%}

//#endif


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
    return _down_cast_map[subclass](tensor)
%}


