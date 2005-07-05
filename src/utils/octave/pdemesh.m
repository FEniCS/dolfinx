function h = pdemesh(p, e, t, u)

% Plots the mesh described by p, e, t (points, edges, triangles).
%
% Copyright (C) 2004-2005 Johan Jansson.
% Licensed under the GNU GPL Version 2.
%
% First added:  2004-01-23
% Last changed: 2005

if(nargin == 3)
  u = zeros(size(p, 2), 1);
end

pdeplot(p, [], t, 'xydata', u)
