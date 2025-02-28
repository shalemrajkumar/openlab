% init Solving Ax=b using Gauss Siedel Method

A = [1 1 1; 2 1 3; 3 4 -2];
b = [4; 7; 9];
Ab = [A b];

% Rearrage the matrix to be diagonally dominant matrix
% 2 -> 3 -> 1

As = [Ab(2, :); Ab(3, :); Ab(1, :)];

% Initializing
n = 3;
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