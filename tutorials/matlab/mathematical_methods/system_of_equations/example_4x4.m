% init Solving Ax=b using Gauss Siedel Method

A = [1 2 2 1; 2 2 4 2; 1 3 2 5; 2 6 5 8];
b = [1; 0; 2; 4];
Ab = [A b];

% Rearrage the matrix to be diagonally dominant matrix
% 2 -> 3 -> 1
%% unable to solve
As = [Ab(1, :); Ab(2, :); Ab(3, :); Ab(4, :)];

% Initializing
n = 4;
x = zeros(n, 1);
err = zeros(n, 1);

% G-S Iterations
for iter = 1: 25
    for k = 1: n
        % to calculate error we store x_old
        x_old = x(k);

        % gauss siedel 
        num = As(k, end) - As(k, 1: k-1) *x(1: k-1) - As(k, k+1:n) * x(k+1:n);
        x(k) = num / As(k, k);

        %error
        err(k) = abs(x(k)  - x_old);
    end

    disp(['Iter', num2str(iter), '; Error = ', num2str(max(err))]);
end