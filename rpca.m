function [L,S] = rpca(X,tau,gamma,max_iter)
    if (~exist('tau', 'var'))
        tau = 1;
    end
    
    if (~exist('gamma', 'var'))
        gamma = 1;
    end
    
    if (~exist('max_iter', 'var'))
        max_iter = 200;
    end
    eps = 1e-8;
    sz = size(X);
    L = X;
    S = zeros(sz);
    k = 10;
    U = zeros(sz); %Lagrangian multiplier
    for i = 1:max_iter
        for j = 1:sz(3)
            L(:,:,j) = shrinkage(k*(X(:,:,j)-S(:,:,j))+  U(:,:,j),tau)/k;
            tmp = k*(X(:,:,j) - L(:,:,j))+U(:,:,j);
            S(:,:,j) = sign(tmp).*max((abs(tmp)-gamma),0)/k;
            U(:,:,j) = U(:,:,j) + k*(X(:,:,j)-L(:,:,j)-S(:,:,j));
        end
        k = k*1.1;
        if norm(X(:)-L(:)-S(:),'fro') < eps
            fprintf('the rpca iteration for testing data is %d\n',i);
            break;     
        end   
    end
end
            
        