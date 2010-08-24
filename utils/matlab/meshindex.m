function index = meshindex(p, t)

% MESHINDEX - COMPUTE APPROXIMATE MULTI-ADAPTIVE EFFICIENCY INDEX FOR MESH
%
% Usage: index = meshindex(p, t)
%
%   p - points    (exported from PDE Toolbox)
%   t - triangles (exported from PDE Toolbox)
%	
% Copyright (C) 2005 Anders Logg.
% Licensed under the GNU LGPL Version 2.1.

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
