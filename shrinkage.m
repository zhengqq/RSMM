function [D nuc rk] = shrinkage(X, tau)
    [U, S, V] = svd(X);
    s = max(0, S-tau);
    nuc = sum(diag(s));
    D = U *  s * V';
    rk = sum(diag(s>0));
end