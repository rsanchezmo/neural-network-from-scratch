%% LOAD THE DATASET
load('data.mat');

% The dataset size is:
%   X [5000,400] -- 5000 examples and 400 fields
%   y [5000,1]   -- 5000 examples and 1 output

m = size(X, 1); % number of examples

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel); % this function is property of machineLearningCourse on Coursera

%% DEFINING THE NEURAL NETWORK ARCHITECTURE

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

layer_sizes = [input_layer_size, 40 ,num_labels];

% we need to create the y_new that is a matrix with size of m*num_labels
y_new = (1:10) == y;

% initialize the weights randomly [the weights have the bias inside on the first column]
for i = 1:(size(layer_sizes, 2)-1)
    initial_weights{i} = randInitializeWeights(layer_sizes(i), layer_sizes(i+1));
end

% Weight regularization parameter and learning rate
lambda = 1;
alfa = 1;

maxIter = 1000;

loss = [];
count = [];
precisionT = [];
weights = initial_weights;
figure;
ax1 = subplot(2,1,1);
ax2 = subplot(2,1,2);

tic; 
for i = 1:maxIter
    [J, weights] = train(weights, layer_sizes,...
                                   X, y_new, lambda, alfa);
    if mod(i,25) == 0
        loss = [loss;J];
        count = [count;i];
        plot(ax1,count, loss, 'LineWidth', 2);
        title(ax1,'COST EVOLUTION');
        xlabel(ax1,'Iterations');
        ylabel(ax1,'Cost function')
        pred = predict(weights, X, layer_sizes);
        precision = mean(double(pred == y)) * 100;
        precisionT = [precisionT;precision];
        plot(ax2,count, precisionT, 'LineWidth', 2);
        title(ax2,'PRECISION EVOLUTION');
        xlabel(ax2,'Iterations');
        ylabel(ax2,'Precision');
        disp(['Iteration #: ' num2str(i) ' / ' num2str(maxIter) ' | Cost J: ' num2str(J) ' | Precission: ' ...
                num2str(precision)]);
        drawnow();
    end
    
end

finT = toc;


disp(['Time spent on training the net: ' num2str(finT) ' seconds' ])


%% test our model on the training set
figure;
i = randi(length(y));
pred = predict(weights, X(i,:), layer_sizes);
imshow(reshape(X(i,:),20,20));
fprintf('True class: %d  |  Predicted class: %d\n',y(i),pred);


