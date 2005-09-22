% Plot solution using MATLAB

% Load solution
stokes

% Pick out velocity and pressure
v = u(:,1:2);
p = u(:,3);

% Plot velocity
figure(1); clf
pdeplot(points, edges, cells, 'flowdata', v)
axis([0 1 0 1])
xlabel('x')
ylabel('y')
title('Stokes velocity')

% Plot pressure
figure(2); clf
pdesurf(points, cells, p)
axis([0 1 0 1])
xlabel('x')
ylabel('y')
title('Stokes pressure')
