function [W,b,L,S,obj_new] = SparLR_ADMM(X,y,c,tau,gamma,max_iter)
    if (~exist('c', 'var'))
        c = 1;
    end
    
    if (~exist('tau', 'var'))
        tau = 1;
    end
    
    if (~exist('gamma', 'var'))
        gamma = 1;
    end
    
    if (~exist('max_iter', 'var'))
        max_iter = 500;
    end
    %initialization for parameters 
    rho = 1;
    lambda = 1;
    L = X; %low rank part
    sz = size(X);
    S = zeros(sz); %sparse part
    %W = zeros(sz(1),sz(2)); %classifier regression matrix
    Z = zeros(sz(1),sz(2)); % addition parameter for W in ADMM
    b = 0; % classifier bias 
    V = zeros(sz(1),sz(2));
    U = zeros(sz);  %d1xd2xn
    n = sz(3);
    eps = 1e-8;
    kappa = 1.1;
    
%     x = reshape(X,[sz(1)*sz(2),sz(3)]); % (d1xd2)xn
    
    inner_iter = 1000;
    obj_new = zeros(max_iter,1);
    obj_old = 10000;
%     figure;
    for iter = 1:max_iter
        l = reshape(L,[sz(1)*sz(2),n]); % (d1xd2) x n
        K = (l'*l).*(y*y')/rho;
        %l = l'; %nx(d1xd2)
        z = reshape(Z,[sz(1)*sz(2),1]);         
        v = reshape(V,[sz(1)*sz(2),1]);
        q = 1 - (l'*(v+rho*z).*y)/rho;
        opt = struct('TolKKT', eps/100, 'MaxIter', inner_iter, 'verb', 0);
        LB = zeros(n,1);
        UB = ones(n,1);
        [alpha,~] = libqp_gsmo(K, -q, y', 0, LB, UB, [], opt);
        w = (v+rho*z+l*(alpha.*y))/rho;  %(d1xd2)x1
        b = sum(y - l' * w)/n;
        W = reshape(w,[sz(1),sz(2)]);
        Z = shrinkage(rho*W - V,c)/rho;
        V = V - rho*(W - Z);
        rho = rho * kappa;
        %Solve for Li
        for i = 1:n
            if y(i)*trace(W'*L(:,:,i)) >= 1
                grad_hinge = 0;
            else
                grad_hinge = -y(i)*W;
            end
            tmp1 = lambda*(X(:,:,i)-S(:,:,i))+U(:,:,i);
%             L(:,:,i) = shrinkage((alpha(i)*y(i)*W+tmp1),tau)/lambda;
            L(:,:,i) = shrinkage((-grad_hinge+tmp1),tau)/lambda;
            tmp = lambda*(X(:,:,i)-L(:,:,i))+U(:,:,i);
            S(:,:,i) = sign(tmp).*max((abs(tmp)-gamma),0)/lambda;
            U(:,:,i) = U(:,:,i) - lambda*(L(:,:,i)+S(:,:,i)-X(:,:,i));
            
        end
        lambda= lambda*kappa;
        obj_new(iter) = objective_function(L,S,y,W,b,c,tau,gamma);
%         if (abs(obj_new(iter) - obj_old) < 1e-7 || obj_new(iter) > obj_old)
        if(abs(obj_new(iter)-obj_old) < eps)
            fprintf('the outer iteration is %d\n', iter);
            break;
        end
        obj_old = obj_new(iter);
%         plot(iter,obj_new(iter),'bo-');
%         hold on;
    end
    
     function z = norm_nuc(X)
        z = sum(svd(X));
     end
    
     function z = norm_l1(X)
        z = sum(X(:));
     end
 
    function obj = objective_function(L,S,y,W,b,c,tau,gamma)
        % L,S: d1 x d2 x n; W: d1 x d2;
        sz0 = size(L);
        l0 = reshape(L,[sz0(1)*sz0(2),sz0(3)]);
        w0 = reshape(W,[sz0(1)*sz0(2),1]);
        tmp_nuc = 0;
        tmp_l1 = 0;
        for k = 1:sz0(3)
            tmp_nuc = tmp_nuc + norm_nuc(L(:,:,k));
            tmp_l1 = tmp_l1 + + norm_l1(S(:,:,k));
        end
        obj = c*norm_nuc(W) + sum(max(1-y.*(l0'*w0+b),0))+ tau*tmp_nuc + gamma*tmp_l1;
%         obj = c*norm_nuc(W) + sum(max(1-y.*(l0'*w0+b),0));
    end
end