% Plot solution using Octave

% Load solution
poisson

% Plot the mesh
figure(1)
pdemesh(points, edges, cells)
title('Mesh')

% Plot with pdesurf
figure(2)
pdesurf(points, cells, u)
title('Poisson')
