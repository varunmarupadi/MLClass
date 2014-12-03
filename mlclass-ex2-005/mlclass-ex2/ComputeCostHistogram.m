function J_vals = ComputeCostHistogram(X, y)
J_vals = ones(100,100);
for i = -100:100
    for j = -100:100
        theta = [i;j];
        v = sigmoid(X * theta) - y;
        J_vals(i+101, j+101) = (v'*v)/2;
    end
end
