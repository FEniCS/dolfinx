//%template(STLVectorDirichletBCPtr) std::vector<dolfin::DirichletBC *>;
//%template(STLVectorBoundaryConditionPtr) std::vector<dolfin::BoundaryCondition *>;
//%template(STLVectorFunctionPtr) std::vector<dolfin::Function *>;
//%template(STLVectorFunctionSpacePtr) std::vector<dolfin::FunctionSpace *>;
//%template(STLVectorUInt) std::vector<dolfin::uint>;
//%template(STLVectorDouble) std::vector<double>;
//%template(STLVectorString) std::vector<std::string>;
//%template(STLPairUInt) std::pair<dolfin::uint,dolfin::uint>;

/*%template(BOOSTUnorderSetUInt) boost::unordered_set<dolfin::uint>;*/

//-----------------------------------------------------------------------------
// Ignore const array interface (Used if the Array type is a const)
//-----------------------------------------------------------------------------
%define CONST_ARRAY_IGNORES(TYPE)
%ignore dolfin::Array<const TYPE>::Array(uint N);
%ignore dolfin::Array<const TYPE>::array();
%ignore dolfin::Array<const TYPE>::resize(uint N);
%ignore dolfin::Array<const TYPE>::zero();
%ignore dolfin::Array<const TYPE>::update();
%ignore dolfin::Array<const TYPE>::__setitem__;
%enddef

//-----------------------------------------------------------------------------
// Modifications of the Array interface
//-----------------------------------------------------------------------------
%define ARRAY_EXTENSIONS(TYPE, TYPENAME, NUMPYTYPE)

// Construct value wrapper for dolfin::Array<TYPE>
// Valuewrapper is used so a return by value Array does not make an extra copy
// in any typemaps
%feature("valuewrapper") dolfin::Array<TYPE>;

%ignore dolfin::Array<TYPE>::Array(uint N, boost::shared_array<TYPE> x);

// Cannot construct an Array from another Array. 
// Use NumPy Array instead
%ignore dolfin::Array<TYPE>::Array(const Array& other);
 
%template(TYPENAME ## Array) dolfin::Array<TYPE>;

%feature("docstring") dolfin::Array::__getitem__ "Missing docstring";
%feature("docstring") dolfin::Array::__setitem__ "Missing docstring";
%feature("docstring") dolfin::Array::array "Missing docstring";
%extend dolfin::Array<TYPE> {
  TYPE __getitem__(unsigned int i) const { return (*self)[i]; }
  void __setitem__(unsigned int i, const TYPE& val) { (*self)[i] = val; }

  PyObject * array(){
    npy_intp dims[1];
    dims[0] = self->size();
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(1, dims, NUMPYTYPE, (char *)(self->data().get())));
    if ( array == NULL ) return NULL;
    return PyArray_Return(array);
  }

}
%enddef

//-----------------------------------------------------------------------------
// Run Array macros, which also instantiate the templates
//-----------------------------------------------------------------------------
CONST_ARRAY_IGNORES(double)
ARRAY_EXTENSIONS(double, Double, NPY_DOUBLE)
ARRAY_EXTENSIONS(const double, ConstDouble, NPY_DOUBLE)
ARRAY_EXTENSIONS(unsigned int, UInt, NPY_UINT)
ARRAY_EXTENSIONS(int, Int, NPY_INT)

//-----------------------------------------------------------------------------
// Add pretty print for Variables
//-----------------------------------------------------------------------------
%feature("docstring") dolfin::Variable::__str__ "Missing docstring";
%extend dolfin::Variable
{
  std::string __str__() const
  {
    return self->str(false);
  }
}
