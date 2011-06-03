function index = meshindex(p, t)

% MESHINDEX - COMPUTE APPROXIMATE MULTI-ADAPTIVE EFFICIENCY INDEX FOR MESH
%
% Usage: index = meshindex(p, t)
%
%   p - points    (exported from PDE Toolbox)
%   t - triangles (exported from PDE Toolbox)
%	
% Copyright (C) 2005 Anders Logg
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

hmin = 1.0;
hlist = hmin*ones(size(p, 2), 1);

for triangle = t

  i0 = triangle(1);
  i1 = triangle(2);
  i2 = triangle(3);

  p0 = p(:, i0);
  p1 = p(:, i1);
  p2 = p(:, i2);

  h0 = norm(p1 - p0);
  h1 = norm(p2 - p1);
  h2 = norm(p0 - p2);

  h = min([h0, h1, h2]);

  hlist(i0) = min(hlist(i0), h);
  hlist(i1) = min(hlist(i1), h);
  hlist(i2) = min(hlist(i2), h);

  hmin = min(h, hmin);

end

index = (length(hlist) / hmin) / sum(1.0 ./ hlist);
