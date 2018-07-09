function [pred,val] = multi_prediction(X,model)
% X is a tensor data: d1 x d2 x num_training
sz = size(X);
val = zeros(sz(3),length(model));
for t = 1:length(model)
    val(:,t) = squeeze(sum(sum(bsxfun(@times,X,model{t}.W)))) + model{t}.b;       
end
val = val';
[~,ind] = max(val);
pred = ind';