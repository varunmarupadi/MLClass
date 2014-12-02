function [training_X, training_y] = GetTrainingValues(X, y, i)
    m = length(y);
    indices = randperm(m, i);
    training_X = X(indices, :);
    training_y = y(indices, :);
end
