function h = pdesurf(p, t, u)

% Plots the scalar solution u on the mesh described by p, t (points,
% triangles).							    
%
% Copyright (C) 2004-2005 Johan Jansson.
% Licensed under the GNU LGPL Version 2.1.
%
% First added:  2004-01-23
% Last changed: 2005

pdeplot(p, [], t, 'xydata', u)
