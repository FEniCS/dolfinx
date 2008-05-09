
%extend dolfin::Function{
const dolfin::Cell& new_cell() const{
    return self->cell();
}
}

