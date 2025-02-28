u = zeros(101, 1);
u(1) = 0;
for i=1: 100
    u(i+1) = f(0, u(i), 0.05);

end

plot(u)

%% function

function result = f(t, u, del_t)
    result = (-1 + sqrt(1 + (12 * del_t)*(u + (6 * del_t)))) / (6 * del_t);
end