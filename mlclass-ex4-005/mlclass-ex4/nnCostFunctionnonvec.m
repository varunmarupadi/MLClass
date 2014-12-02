function [J, grad] = nnCostFunctionnonvec(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = max(y);
         
% You need to return the following variables correctly 
sum_term = 0;
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for i = 1:m
    X_term = [1; X(i,:)'];
    z2 = Theta1 * X_term;
    a2 = [1; sigmoid(z2)];
    hypothesis = sigmoid(Theta2 * a2);
    
    y_term = zeros(K,1);
    y_term(y(i)) = 1;
    
    term1 = y_term .* log(hypothesis);
    term2 = (1-y_term) .* log(1 - hypothesis); 
    sum_term = sum_term + sum(term1 + term2);
    
    d_3 = hypothesis - y_term;    
    delta_2 = delta_2 + d_3 * a2';

    d_2 = Theta2' * d_3 .* (a2 .* (1-a2));
    d_2 = d_2(2:end);
    delta_1 = delta_1 + d_2 * X_term';
end

Theta1_d = Theta1;
Theta1_d(:,1) = zeros;
Theta2_d = Theta2;
Theta2_d(:,1) = zeros;

J = -sum_term/m + (sum(sum(Theta1_d.^2)) + sum(sum(Theta2_d.^2)))*(lambda/m/2);
Theta1_grad = delta_1/m + (lambda/m)*Theta1_d;
Theta2_grad = delta_2/m + (lambda/m)*Theta2_d;

grad = [Theta1_grad(:) ; Theta2_grad(:)];