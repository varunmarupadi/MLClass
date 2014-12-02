function k = PreserveVariance(S, variance)
% returns the k features that preserver 'variance' percentage (0-100) of
% the variance of U, where S is the diagonal singular value matrix obtained
% from the singular value decomposition of U.

Sigma = sum(S);
desired = sum(Sigma) * variance / 100;
k = find(cumsum(Sigma) > desired, 1);