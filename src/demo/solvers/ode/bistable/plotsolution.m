% Copyright (C) 2002 Johan Hoffman and Anders Logg.
% Licensed under the GNU GPL Version 2.

% Load mesh and solution
mesh
primal

% Make the movie
clf
clear M
for n = 1:size(u,2)
  
  pdesurf(points, cells, u(:,n))
  xlabel('x')
  ylabel('y')
  %shading interp
  shading faceted
  colormap hot
  view(2)

  M(n) = getframe;

  disp('Press any key to continue...')
  pause
  
end

movie(M)
