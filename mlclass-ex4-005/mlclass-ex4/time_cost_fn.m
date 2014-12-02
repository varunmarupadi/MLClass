function [total_elapsed_vec, total_elapsed_nonvec] = time_cost_fn(X,y) 
total_elapsed_vec = zeros(size(clock)); 
total_elapsed_nonvec = zeros(size(clock));
for i = 1:100
    initial_Theta1 = randInitializeWeights(400, 25);
    initial_Theta2 = randInitializeWeights(25, 10);

    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    
    % Vectorized
    c = clock;
    [~,~] = nnCostFunction(initial_nn_params, 400, 25, 10, X, y, 10);
    total_elapsed_vec = total_elapsed_vec + clock - c;
    
    % Non vectorized
    c = clock;
    [~,~] = nnCostFunctionnonvec(initial_nn_params, 400, 25, 10, X, y, 10);
    total_elapsed_nonvec = total_elapsed_nonvec + clock - c;
end
    