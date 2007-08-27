// Rename Function --> cpp_Function (gets mapped in assembly.py)
%rename(cpp_Function) dolfin::Function;
%rename(sub) dolfin::Function::operator[];
