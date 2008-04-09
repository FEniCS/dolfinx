// Define uBlas matrix types (typedefs are ignored)
%template(uBlasSparseMatrix) dolfin::uBlasMatrix<dolfin::ublas_sparse_matrix>;
%template(uBlasDenseMatrix) dolfin::uBlasMatrix<dolfin::ublas_dense_matrix>;


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

%extend dolfin::PETScMatrix {
  %pythoncode %{
    def __mul__(self, other):
      v = PETScVector(self.size(0))
      self.mult(other, v)
      return v

  %}
}

%extend dolfin::uBlasMatrix {
  %pythoncode %{
    def __mul__(self, other):
      v = PETScVector(self.size(0))
      self.mult(other, v)
      return v

  %}
}




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
  %}
}



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
  %}
}

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
  %}
}




