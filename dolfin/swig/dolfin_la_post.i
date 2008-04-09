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

%newobject  dolfin::Matrix::__mul__;
%extend dolfin::Matrix {
      dolfin::Vector* __mul__(const dolfin::Vector& x) {
        dolfin::Vector* Ax = new dolfin::Vector((*self).size(0)); 
      (*self).mult(x, *Ax); 
      return Ax;  
    }


}

%extend dolfin::Vector {
    void __iadd__(dolfin::Vector& v) {
      (*self).add(v); 
    }
    void __isub__(dolfin::Vector& v) {
      (*self).add(v,-1.0); 
    }

    void __imul__(real a) {
      (*self).mult(a); 
    }
}


