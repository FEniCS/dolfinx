// Ignore reference (to FunctionSpaces) constructors of Function
%ignore dolfin::Function::Function(const FunctionSpace&);
%ignore dolfin::Function::Function(const FunctionSpace&, GenericVector&);
%ignore dolfin::Function::Function(const FunctionSpace&, std::string);

// Modifying the interface of Function
%rename(_sub) dolfin::Function::operator[];
%rename(assign) dolfin::Function::operator=;
%rename(_in) dolfin::Function::in;

// Modifying the interface of DiscreteFunction
%rename(_sub)    dolfin::DiscreteFunction::operator[];
%rename(assign) dolfin::DiscreteFunction::operator=;
%rename(_in)    dolfin::DiscreteFunction::in;

// Rename eval(val, data) function
%rename(eval_data) dolfin::Function::eval(double* values, const Data& data) const;

// Ignore the Data.x, pointer to the coordinates in the Data object
%ignore dolfin::Data::x;
%rename (x) dolfin::Data::x_();

