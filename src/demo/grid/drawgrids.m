% Copyright (C) 2003 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% Draw grids from the 2D grid refinement example.

disp('Loading unrefined grid')
grid2D_unrefined
p0 = points;
e0 = edges;
t0 = cells;

disp('Loading refined grid')
grid2D_refined
p1 = points;
e1 = edges;
t1 = cells;

clf

subplot(1,2,1)
pdemesh(p0, e0, t0)
axis([0 1 0 1])

subplot(1,2,2)
pdemesh(p1, e1, t1)
axis([0 1 0 1])
