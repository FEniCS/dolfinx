// Instantiate uBLAS matrix types
%template(uBLASSparseMatrix) dolfin::uBLASMatrix<dolfin::ublas_sparse_matrix>;
%template(uBLASDenseMatrix) dolfin::uBLASMatrix<dolfin::ublas_dense_matrix>;
// Define names for uBLAS matrix types
%typedef dolfin::uBLASMatrix<dolfin::ublas_sparse_matrix> uBLASSparseMatrix;
%typedef dolfin::uBLASMatrix<dolfin::ublas_dense_matrix> uBLASDenseMatrix;


#ifdef HAS_SLEPC
%extend dolfin::SLEPcEigenSolver {

PyObject* getEigenvalue(const int emode) {
    dolfin::real err, ecc;
    self->getEigenvalue(err, ecc, emode);

    PyObject* result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(err));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(ecc));
    Py_INCREF(result);
    return result;
}

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


%define LA_POST_INTERFACE(VEC_TYPE,MAT_TYPE)
%extend dolfin::VEC_TYPE
{
  void _scale(dolfin::real a)
  {
    (*self)*=a;
  }
  
  %pythoncode
  %{
    def array(self):
        " Return a numpy array representation of Vector"
        import numpy
        v = numpy.zeros(self.size())
        self.get(v)
        return v

    def __add__(self,other):
        """x.__add__(y) <==> x+y"""
        if isinstance(other,type(self)):
            ret = self.copy()
            ret.axpy(1.0,other)
            return ret
        return NotImplemented
    
    def __sub__(self,other):
        """x.__sub__(y) <==> x-y"""
        if isinstance(other,type(self)):
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
        return NotImplemented
    
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
        if isinstance(other,type(self)):
            self.axpy(1.0,other)
            return self
        return NotImplemented
    
    def __isub__(self,other):
        """x.__isub__(y) <==> x-y"""
        if isinstance(other,type(self)):
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

%extend dolfin::MAT_TYPE
{
  void _scale(dolfin::real a)
  {
    (*self)*=a;
  }
  
  %pythoncode
  %{
    def array(self):
        " Return a numpy array representation of Matrix"
        import numpy
        A = numpy.zeros((self.size(0), self.size(1)))
        c = ArrayUInt()
        v = ArrayDouble()
        for i in xrange(self.size(0)):
            self.getrow(i, c, v)
            A[i,c] = v
        return A

    def __add__(self,other):
        """x.__add__(y) <==> x+y"""
        if isinstance(other,type(self)):
            ret = self.copy()
            ret.axpy(1.0,other)
            return ret
        return NotImplemented
    
    def __sub__(self,other):
        """x.__sub__(y) <==> x-y"""
        if isinstance(other,type(self)):
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
        elif isinstance(other,GenericVector):
            ret = other.copy()
            self.mult(other, ret)
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
        return NotImplemented
    
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
        if isinstance(other,type(self)):
            self.axpy(1.0,other)
            return self
        return NotImplemented
    
    def __isub__(self,other):
        """x.__isub__(y) <==> x-y"""
        if isinstance(other,type(self)):
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

LA_POST_INTERFACE(GenericVector,GenericMatrix)
LA_POST_INTERFACE(Vector,Matrix)
LA_POST_INTERFACE(uBLASVector,uBLASMatrix)

#ifdef HAS_PETSC
LA_POST_INTERFACE(PETScVector,PETScMatrix)
#endif

#ifdef HAS_TRILINOS
LA_POST_INTERFACE(EpetraVector,EpetraMatrix)
#endif

#ifdef HAS_MTL4
LA_POST_INTERFACE(MTL4Vector,MTL4Matrix)
#endif



//%extend dolfin::GenericVector
//{
//  %pythoncode
//  %{
//    def array(self):
//        import numpy
//        v = numpy.zeros(self.size())
//        self.get(v)
//        return v
//  %}
//}
//
//%extend dolfin::GenericMatrix
//{
//  %pythoncode
//  %{
//    def array(self):
//        import numpy
//        A = numpy.zeros((self.size(0), self.size(1)))
//        c = ArrayUInt()
//        v = ArrayDouble()
//        for i in xrange(self.size(0)):
//            self.getrow(i, c, v)
//            A[i,c] = v
//        return A
//  %}
//}
//
//%extend dolfin::Matrix {
//  %pythoncode %{
//    def __mul__(self, other):
//      v = Vector(self.size(0))
//      self.mult(other, v)
//      return v
//
//  %}
//}
//
//#ifdef HAS_PETSC
//%extend dolfin::PETScMatrix {
//  %pythoncode %{
//    def __mul__(self, other):
//      v = PETScVector(self.size(0))
//      self.mult(other, v)
//      return v
//
//  %}
//}
//#else
//%extend dolfin::uBLASMatrix {
//  %pythoncode %{
//    def __mul__(self, other):
//      v = PETScVector(self.size(0))
//      self.mult(other, v)
//      return v
//
//  %}
//}
//#endif
//#ifdef HAS_TRILINOS
//%extend dolfin::EpetraMatrix {
//  %pythoncode %{
//    def __mul__(self, other):
//      v = EpetraVector(self.size(0))
//      self.mult(other, v)
//      return v
//
//  %}
//}
//#endif
//
//
//%extend dolfin::Vector {
//  %pythoncode %{
//
//    def __add__(self, v): 
//      a = self.copy() 
//      a += v
//      return a
//
//    def __sub__(self, v): 
//      a = self.copy() 
//      a -= v
//      return a
//
//    def __mul__(self, v): 
//      a = self.copy() 
//      a *= v
//      return a
//
//    def __rmul__(self, v):
//      return self.__mul__(v)
//  %}
//}
//
//
//
//#ifdef HAS_PETSC
//%extend dolfin::PETScVector {
//  %pythoncode %{
//
//    def __add__(self, v): 
//      a = self.copy() 
//      a += v
//      return a
//
//    def __sub__(self, v):
//      a = self.copy() 
//      a -= v
//      return a
//
//    def __mul__(self, v): 
//      a = self.copy() 
//      a *= v
//      return a
//
//    def __rmul__(self, v):
//      return self.__mul__(v)
//  %}
//}
//#else
//%extend dolfin::uBLASVector {
//  %pythoncode %{
//
//    def __add__(self, v):
//      a = self.copy() 
//      a += v
//      return a
//
//    def __sub__(self, v):
//      a = self.copy() 
//      a -= v
//      return a
//
//    def __mul__(self, v): 
//      a = self.copy() 
//      a *= v
//      return a
//
//    def __rmul__(self, v):
//      return self.__mul__(v)
//  %}
//}
//#endif
//
//#ifdef HAS_TRILINOS
//%extend dolfin::EpetraVector {
//  %pythoncode %{
//
//    def __add__(self, v): 
//      a = self.copy() 
//      a += v
//      return a
//
//    def __sub__(self, v):
//      a = self.copy() 
//      a -= v
//      return a
//
//    def __mul__(self, v): 
//      a = self.copy() 
//      a *= v
//      return a
//
//    def __rmul__(self, v):
//      return self.__mul__(v)
//  %}
//}
//#endif


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



// Initialize tensor type map
%pythoncode %{
_has_type_map = {}
_down_cast_map = {}
%}


// Add uBLAS support

%inline %{
bool has_type_ublas_vector(dolfin::GenericTensor & tensor)
{ return tensor.has_type<dolfin::uBLASVector>(); }

bool has_type_ublas_matrix(dolfin::GenericTensor & tensor)
{ return tensor.has_type<uBLASSparseMatrix>(); }

dolfin::uBLASVector & down_cast_ublas_vector(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::uBLASVector>(); }

uBLASSparseMatrix   & down_cast_ublas_matrix(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<uBLASSparseMatrix>(); }
%}

%pythoncode %{
_has_type_map[uBLASVector] = has_type_ublas_vector
_has_type_map[uBLASSparseMatrix] = has_type_ublas_matrix
_down_cast_map[uBLASVector] = down_cast_ublas_vector
_down_cast_map[uBLASSparseMatrix] = down_cast_ublas_matrix
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


// Add Trilinos support
#ifdef HAS_TRILINOS

%inline %{

bool has_type_epetra_vector(dolfin::GenericTensor & tensor)
{ return tensor.has_type<dolfin::EpetraVector>(); }

bool has_type_epetra_matrix(dolfin::GenericTensor & tensor)
{ return tensor.has_type<dolfin::EpetraMatrix>(); }

dolfin::EpetraVector & down_cast_epetra_vector(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::EpetraVector>(); }

dolfin::EpetraMatrix & down_cast_epetra_matrix(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::EpetraMatrix>(); }
%}

%pythoncode %{
_has_type_map[EpetraVector] = has_type_epetra_vector
_has_type_map[EpetraMatrix] = has_type_epetra_matrix
_down_cast_map[EpetraVector] = down_cast_epetra_vector
_down_cast_map[EpetraMatrix] = down_cast_epetra_matrix
%}

#endif

// Add MTL4 support
#ifdef HAS_MTL4

%inline %{

bool has_type_mtl4_vector(dolfin::GenericTensor & tensor)
{ return tensor.has_type<dolfin::MTL4Vector>(); }

bool has_type_mtl4_matrix(dolfin::GenericTensor & tensor)
{ return tensor.has_type<dolfin::MTL4Matrix>(); }

dolfin::MTL4Vector & down_cast_mtl4_vector(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::MTL4Vector>(); }

dolfin::MTL4Matrix & down_cast_mtl4_matrix(dolfin::GenericTensor & tensor)
{ return tensor.down_cast<dolfin::MTL4Matrix>(); }
%}

%pythoncode %{
_has_type_map[MTL4Vector] = has_type_mtl4_vector
_has_type_map[MTL4Matrix] = has_type_mtl4_matrix
_down_cast_map[MTL4Vector] = down_cast_mtl4_vector
_down_cast_map[MTL4Matrix] = down_cast_mtl4_matrix
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
    return _down_cast_map[subclass](tensor)
%}


