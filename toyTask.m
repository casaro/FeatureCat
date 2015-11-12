function [results] = toyTask(D, E, W, s1, s2, s1_test, s2_test)
    
    %% With Training
    B = (D * E * W')';
    mean1 = mean(B(s1,:), 1);
    mean2 = mean(B(s2,:), 1);
    acc = 0;
    testsize = 0;
    cosine1 = pdist2(B(s1_test,:),mean1,'euclidean');
    cosine2 = pdist2(B(s1_test,:),mean2,'euclidean');
    acc = acc + sum(cosine1 < cosine2);
    testsize = testsize + length(cosine1 < cosine2);   
    cosine1 = pdist2(B(s2_test,:),mean1,'euclidean');
    cosine2 = pdist2(B(s2_test,:),mean2,'euclidean');
    acc = acc + sum(cosine1 > cosine2);
    testsize = testsize + length(cosine1 < cosine2);  
    fprintf('Result:   %4.3f (%d/%d)\n', acc/testsize, acc, testsize);
    
    %% Without Training
    B = (D * W')';
    mean1 = mean(B(s1,:), 1);
    mean2 = mean(B(s2,:), 1);
    acc_org = 0;
    testsize = 0;
    cosine1 = pdist2(B(s1_test,:),mean1,'euclidean');
    cosine2 = pdist2(B(s1_test,:),mean2,'euclidean');
    acc_org = acc_org + sum(cosine1 < cosine2);
    testsize = testsize + length(cosine1 < cosine2);   
    cosine1 = pdist2(B(s2_test,:),mean1,'euclidean');
    cosine2 = pdist2(B(s2_test,:),mean2,'euclidean');
    acc_org = acc_org + sum(cosine1 > cosine2);
    testsize = testsize + length(cosine1 < cosine2);  
    fprintf('Original: %4.3f (%d/%d)\n', acc_org/testsize, acc_org, testsize);
    
    %% With SVD
    train_size = size(D, 1);
    [~, ~, V_svd] = svd(W, 'econ');
    B = W * V_svd(:,1:train_size);
    mean1 = mean(B(s1,:), 1);
    mean2 = mean(B(s2,:), 1);
    acc_svd = 0;
    testsize = 0;
    cosine1 = pdist2(B(s1_test,:),mean1,'euclidean');
    cosine2 = pdist2(B(s1_test,:),mean2,'euclidean');
    acc_svd = acc_svd + sum(cosine1 < cosine2);
    testsize = testsize + length(cosine1 < cosine2);   
    cosine1 = pdist2(B(s2_test,:),mean1,'euclidean');
    cosine2 = pdist2(B(s2_test,:),mean2,'euclidean');
    acc_svd = acc_svd + sum(cosine1 > cosine2);
    testsize = testsize + length(cosine1 < cosine2);  
    fprintf('SVD:      %4.3f (%d/%d)\n', acc_svd/testsize, acc_svd, testsize);
    
    results = [acc acc_org acc_svd testsize];
    
end

