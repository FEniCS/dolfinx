% Load solution
poisson

% Reference value for the error
Eref = 0.320322382100906;

% Compute exact solution
x = points(1,:)';
y = points(2,:)';
z = points(3,:)';

uu = sin(pi*x) .* sin(2.0*pi*y) .* sin(3.0*pi*z);

% Compute error
e = u - uu;
E = max(abs(e));

disp(['Size of error: ' num2str(E)])
%disp(['Should be:     ' num2str(Eref)])
