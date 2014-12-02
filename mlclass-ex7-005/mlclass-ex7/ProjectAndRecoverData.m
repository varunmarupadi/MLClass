function X_recovered_pca = ProjectAndRecoverData(X_recovered_pca, U, d, mu, sigma)
    X_reduced = projectData(X_recovered_pca, U, d);
    X_recovered_pca = recoverData(X_reduced, U, d);
    X_recovered_pca = bsxfun(@times, X_recovered_pca, sigma);
    X_recovered_pca = bsxfun(@plus, X_recovered_pca, mu);
    % clip values to stay between [0,1]. Maybe there's a better way to do
    % this?
    %X_recovered_pca(X_recovered_pca > 1) = 1.00;
    %X_recovered_pca(X_recovered_pca < 0) = 0.00;
    
    % Alternative: push everything to +ve, normalize to max 1
    X_recovered_pca = X_recovered_pca - min(min(X_recovered_pca));
    X_recovered_pca = X_recovered_pca / max(max(X_recovered_pca));
    X_recovered_pca(X_recovered_pca > 1)
    X_recovered_pca(X_recovered_pca < 0)