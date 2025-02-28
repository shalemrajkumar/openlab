%% question : T'' = gamma(T-25) ; gamma * h^2 is alpha T(0) = 100, T(1) = 25 b = beta
% Code for Tri-Diagonal Matrix Algorithm
% To solve Ax = b
% where A has tri-diagonal structure with sub-diagonal, diagonal, super-diagonal represented by l, d, u

n = 10;
alpha = 0.04;
beta = -1;

%% Creating the problem matrices

l(1,1) = 0; l(n+1,1) = 0;
u (1,1) = 0; u(n+1,1) = 0;
d (1,1) = 1; d(n+1,1) = 1;
b (1,1) = 100; b(n+1,1) = 25;

l(2:n,1) = 1;
u(2:n,1) = 1;
d (2:n,1) = -(2+alpha);
b (2:n,1) = beta;

%% Thomas Algorithm
for i = 1:n
    % Normalize by dividing with d(i)
    u(i) = u(i)/d(i);
    b(i) = b(i)/d(i);
    d(i) = 1;

    % Using pivot element for elimination
    alpha = l(i+1);
    l(i+1) = 0;
    d(i+1) = d(i+1) - alpha*u(i);
    b(i+1) = b(i+1) - alpha*b(i);
end

%% Back-substitution
x = zeros(n+1,1) ;
x(n+1,1) = b(n+1) /d (n+1) ;

for i = n:-1:1
    x(i) = b(i) - u(i)*x(i+1);
end