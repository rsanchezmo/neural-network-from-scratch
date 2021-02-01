function p = predict(weights, X, layer_sizes)


m = size(X, 1);


% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% initialize the sum and the activations
for i = 1:(size(layer_sizes,2))
    z{i} = zeros(m,layer_sizes(i));
    a{i} = zeros(m,layer_sizes(i));
end

a{1,1} = [ones(m, 1) X]; 

for i = 2:(size(layer_sizes,2))
    z{1,i} = a{1,i-1}*weights{1,i-1}';
    a{1,i} = sigmoid(z{1,i});
    if i ~= size(layer_sizes,2)
        a{1,i} = [ones(m,1) a{1,i}];
    end
end

[~, p] = max(a{1,end}, [], 2);


end
