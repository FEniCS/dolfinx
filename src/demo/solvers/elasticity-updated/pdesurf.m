function h = pdesurf(p, t, u)

% Plots the scalar solution u on the mesh described by p, t (points,
% triangles).							    
%
% Copyright (C) 2004 Johan Jansson.
% Licensed under the GNU GPL Version 2.

  pdeplot(p, [], t, 'xydata', u)
