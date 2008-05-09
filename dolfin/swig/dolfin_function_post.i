// Trick to expose protected member cell() in Python
%extend dolfin::Function{
const dolfin::Cell& new_cell() const{
    return self->cell();
}
}

// Trick to expose protected member normal() in Python
%extend dolfin::Function{
dolfin::Point new_normal() const{
    return self->normal();
}
}
