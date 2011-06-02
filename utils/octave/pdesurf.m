function h = pdesurf(p, t, u)

% Plots the scalar solution u on the mesh described by p, t (points,
% triangles).							    
%
% Copyright (C) 2004-2005 Johan Jansson
%
% This file is part of DOLFIN.
%
% DOLFIN is free software: you can redistribute it and/or modify
% it under the terms of the GNU Lesser General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% DOLFIN is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
%
% First added:  2004-01-23
% Last changed: 2005

pdeplot(p, [], t, 'xydata', u)
