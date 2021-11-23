function [net,errs] = learn(net, data_train, labels_train, data_test, labels_test)

Tk = net.iterations;              % number of classifiers to generate
K = length(data_train);           % number of data sets 
net.classifiers = cell(Tk*K, 1);  % cell array with total number of classifiers
net.beta = zeros(Tk*K, 1);        % beta will set the classifier weights
c_count = 0;              % keep track of the number of classifiers at each time
errs = zeros(Tk*K, 1);    % prediction errors on the test data set
disp(data_train)
for k = 1:K  
  

  data_train_k = data_train{k};
  labels_train_k = labels_train{k};
  D = ones(numel(labels_train_k), 1)/numel(labels_train_k);

  if k > 1
    predictions = classify_ensemble(net, data_train_k, labels_train_k,...
			c_count);   % predict on the training data
    epsilon_kt = sum(D(predictions ~= labels_train_k)); % error on D
    beta_kt = epsilon_kt/(1-epsilon_kt);                % normalized error on D
    D(predictions == labels_train_k) = beta_kt * D(predictions == labels_train_k);
  end
  
  for t = 1:Tk
    c_count = c_count + 1;
    
    D = D / sum(D);
    
    index = randsample(1:numel(D), numel(D), true, D);

    net.classifiers{c_count} = classifier_train(...
      net.base_classifier, ...
      data_train_k(index, :), ...
      labels_train_k(index));
  
    y = classifier_test(net.classifiers{c_count}, data_train_k);
    epsilon_kt = sum(D(y ~= labels_train_k));
    net.beta(c_count) = epsilon_kt/(1-epsilon_kt);
    

    predictions = classify_ensemble(net, data_train_k, labels_train_k, ...
      c_count);
    E_kt = sum(D(predictions ~= labels_train_k));
    if E_kt > 0.5 
      E_kt = 0.5;   
    end
     
    Bkt = E_kt / (1 - E_kt);
    D(predictions == labels_train_k) = Bkt * D(predictions == labels_train_k);
    D = D / sum(D);
    
    [predictions,posterior] = classify_ensemble(net, data_test, ...
      labels_test, c_count);
    errs(c_count) = sum(predictions ~= labels_test)/numel(labels_test); 
  end
  
  
end



function [predictions,posterior] = classify_ensemble(net, data, labels, lims)
n_experts = lims;
weights = log(1./net.beta(1:lims));
p = zeros(numel(labels), net.mclass);
for k = 1:n_experts
  y = classifier_test(net.classifiers{k}, data);
  
  % this is inefficient, but it does the job 
  for m = 1:numel(y)
    p(m,y(m)) = p(m,y(m)) + weights(k);
  end
end
[~,predictions] = max(p');
predictions = predictions';
posterior = p./repmat(sum(p,2),1,net.mclass);
