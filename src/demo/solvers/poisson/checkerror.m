% Load solution
poisson

format long

% Reference value for the error
Eref = 0.0498860445657127;

% Compute exact solution
x  = points(1,:)';
y  = points(2,:)';
uu = sin(pi*x);

% Compute error
e = u - uu;
E = max(abs(e));

disp(['Size of error: ' num2str(E)])
disp(['Should be:     ' num2str(Eref)])
