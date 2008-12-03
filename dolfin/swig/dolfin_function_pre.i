// Modifying the interface of Function
%rename(sub) dolfin::Function::operator[];
%rename(assign) dolfin::Function::operator=;
%rename(_in) dolfin::Function::in;

// Modifying the interface of DiscreteFunction
%rename(sub)    dolfin::DiscreteFunction::operator[];
%rename(assign) dolfin::DiscreteFunction::operator=;
%rename(_in)    dolfin::DiscreteFunction::in;

// Ignore eval(val, data) function
%ignore dolfin::Function::eval(double* values, const Data& data) const;
