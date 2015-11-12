function [] = FeatureCat5(varargin)

    %wordsFilename = '/mounts/data/proj/sascha/corpora/GoogleNews-vectors-negative300.bin';
    %wordsFilename = '/mounts/data/proj/fadebac/sa/data_manipulated/corpora/events2012_twitter-semeval2015/mikolov/2015-01-23-train_mikolov-no_elongated-skip-50.sh/skip-50';
    wordsFilename = '/mounts/data/proj/fadebac/sa/data_manipulated/corpora/events2012_twitter-semeval2015/mikolov/2015-05-18-train_mikolov-no_elongated-skip-300.sh/skip-300';
    %wordsFilename = '/mounts/data/proj/sascha/corpora/GoogleNews-vectors-negative300_lower.txt';
    %wordsFilename = '/mounts/data/proj/sascha/corpora/GloVe/glove.twitter.27B.50d.txt';
    
    if any(strfind(wordsFilename, '.bin'))
        [W, dictW] = loadBinaryFile(wordsFilename, 30000);
    else
        [W, dictW] = loadTxtFile(wordsFilename);
        W(:,all(isnan(W),1)) = [];
    end
    
    %% Existing Results
    saveResults = false;
    loadResults = false;
    resultFilename = 'results_twitter_s140_best.mat';
    Results = [];
    if (loadResults && exist(resultFilename, 'file'))
        load(resultFilename);
    end
    
    %% Misc
    s = RandStream('mt19937ar','Seed',0);
    RandStream.setGlobalStream(s);
    
    %% Define Files
    train_lexicons = {...
        '/mounts/data/proj/sascha/corpora/Sentiment_Lexicon/WilWieHof05.txt'
        %'/mounts/data/proj/sascha/corpora/Sentiment_Lexicon/HuLiu04.txt'
        '/mounts/data/proj/sascha/corpora/Sentiment_Lexicon/NRC-Emotion-Lexicon.txt'
        %'/mounts/data/proj/sascha/corpora/Sentiment_Lexicon/NRC-Hashtag-Sentiment-Lexicon.txt'
        %'/mounts/data/proj/sascha/corpora/Sentiment_Lexicon/Sentiment140-Lexicon.txt'
        };  
    fallback_lexicon = '/mounts/data/proj/sascha/corpora/Sentiment_Lexicon/baseline.txt';    
    test_lexicon = '/mounts/data/proj/sascha/corpora/Sentiment_Lexicon/HuLiu04.txt';
  
    %% Load Trainings Lexicon
    dictS_train = [];
    polS_train = [];
    for i=1:length(train_lexicons)
        fileID = fopen(train_lexicons{i});
        Table = textscan(fileID, '%s\t%f\n', 'CollectOutput',1);
        dictS_train = [dictS_train ; Table{1,1}(:, 1)];
        polS_train = [polS_train ; Table{1,2}(:, 1)];
        fclose(fileID);
    end 
    [S, s] = getVectors(dictS_train, W, dictW);
    fprintf('Sentiment Lexicon training size: %d/%d\n', sum(s~=0), length(s));
    %fprintf('Last word found on index %d\n', max(s));
    S(s == 0,:) = [];
    dictS_train(s == 0,:) = [];
    polS_train(s == 0,:) = [];
    ordering = randsample(size(S, 1),size(S, 1));
    S = S(ordering,:);
    dictS_train = dictS_train(ordering,:);
    polS_train = polS_train(ordering,:);
    
    %% Load Fallback Lexicon
    fileID = fopen(fallback_lexicon);
    Table = textscan(fileID, '%s\t%f\n', 'CollectOutput',1);
    dictFB = Table{1,1}(:, 1);
    polFB = Table{1,2}(:, 1);
    fclose(fileID);
    polFB = polFB / 10;
    
    %% Load Test Lexicon
    fileID = fopen(test_lexicon);
    Table = textscan(fileID, '%s\t%f\n', 'CollectOutput',1);
    dictS_test = Table{1,1}(:, 1);
    polS_test = double(Table{1,2}(:, 1));
    fclose(fileID);
    [S_test, id] = getVectors(dictS_test, W, dictW);
    id(ismember(id, s)) = 0; 
    fprintf('Sentiment Lexicon test size: %d/%d\n', sum(id~=0), length(id));
    S_test(id == 0,:) = [];
    polS_test(id == 0,:) = [];
    clearvars id;
      
    %% 
    S_pos_test = S_test(polS_test > 0.5,:);
    S_neg_test = S_test(polS_test < -0.5,:);
    
    num_iters = 1500;
    batchsize = 10;
    
    %% Define Weighting
    Weights = [];
    for a=0:0.1:1
        for b=0.0:0.1:(1-a)
            c = round((1 - a - b) * 100) / 100;
            Weights = [Weights; a b c -0.13 length(dictS_train)];
        end
    end    
    Weights = [0.1 0.5 0.4 -0.13 length(dictS_train)];

    for w=1:size(Weights, 1)
        
        %% Set Weights
        a = Weights(w,1);
        b = Weights(w,2);
        c = Weights(w,3);
        x = Weights(w,4); 
        fprintf('Weighting: %2.1f %2.1f %2.1f\n', a, b, c);
        
        %% Cut to Lexicon Size
        lex_size = Weights(w,5);
        S_ava = S(1:lex_size,:);
        polS_ava = polS_train(1:lex_size,:);
        S_pos = S_ava(polS_ava > 0.5,:);
        S_neg = S_ava(polS_ava < -0.5,:);
        S_neu = S_ava(polS_ava == 0,:);
        while (~isempty(S_pos) && size(S_pos, 1) < 3*batchsize)
            S_pos = [S_pos; S_pos];
        end
        while (~isempty(S_neg) && size(S_neg, 1) < 3*batchsize)
            S_neg = [S_neg; S_neg];
        end
        while (~isempty(S_neu) && size(S_neu, 1) < batchsize)
            S_neu = [S_neu; S_neu];
        end        

        %% Train
        for train_size=[1]

            dim = size(W,2);
            E = eye(dim);
            D = eye(dim);

            s = RandStream('mt19937ar','Seed',0);
            RandStream.setGlobalStream(s);

            fprintf('Size of trained part: %d\n', train_size);
            D_train = [D(1:train_size,:);zeros(dim-train_size,size(D,2))];
            D_train = sparse(D_train);
            %D_one = zeros(dim, dim);
            %D_one(1,1:train_size) = 1;
            D_one = zeros(dim, dim+1);            
            D_one(1,1:train_size+1) = 1;            
            D_one = sparse(D_one);

            [J_history, E, D_one] = train(a, b, c, E, W, train_size, D_train, D_one, num_iters, batchsize, S_pos, S_neg, S_neu, S_ava, polS_train);
            
            Results = evaluate(Results, a, b, c, x, E, W, dictW, train_size, lex_size, D_train, D_one, S_pos, S_neg, S_pos_test, S_neg_test, polFB, dictFB);
            if (saveResults)
                save(resultFilename, 'Results');
            end
            
            plotGraph(J_history, num_iters);
        end
    end
end

function [J_history, E, D_one] = train(a, b, c, E, W, train_size, D_train, D_one, num_iters, batchsize, S_pos, S_neg, S_neu, S_ava, polS_train)

    learningRate = 5;
    dim = size(W,2);    
    J_history = zeros(num_iters,5);    
    
    for iter=1:num_iters

        samplePos = S_pos(randsample(1:size(S_pos, 1),3*batchsize),:);
        sampleNeg = S_neg(randsample(1:size(S_neg, 1),3*batchsize),:);
        V = (samplePos - sampleNeg)';

        [J1, grad1_E] = gradient(V, E, D_train, zeros(size(V)));
        J1 = mean(J1);

        samplePos1 = S_pos(randsample(1:size(S_pos, 1),batchsize),:);
        sampleNeg1 = S_neg(randsample(1:size(S_neg, 1),batchsize),:);
        samplePos2 = S_pos(randsample(1:size(S_pos, 1),batchsize),:);
        sampleNeg2 = S_neg(randsample(1:size(S_neg, 1),batchsize),:);
        if isempty(S_neu)
            sampleNeu1 = [];
            sampleNeu2 = [];
        else
            sampleNeu1 = S_neu(randsample(1:size(S_neu, 1),batchsize),:);
            sampleNeu2 = S_neu(randsample(1:size(S_neu, 1),batchsize),:);
        end
        V = [(samplePos1 - samplePos2) ; (sampleNeg1 - sampleNeg2); (sampleNeu1 - sampleNeu2)]';

        [J2, grad2_E] = gradient(V, E, D_train, zeros(size(V)));
        J2 = mean(J2);

        samplePos1 = S_pos(randsample(1:size(S_pos, 1),batchsize),:);
        sampleNeg1 = S_neg(randsample(1:size(S_neg, 1),batchsize),:);
        samplePos2 = 1 .* [ones(batchsize,train_size) zeros(batchsize, dim-train_size)];
        sampleNeg2 = -1 .* [ones(batchsize,train_size) zeros(batchsize, dim-train_size)];
        if isempty(S_neu)
            sampleNeu1 = [];
            sampleNeu2 = [];
        else
            sampleNeu1 = S_neu(randsample(1:size(S_neu, 1),batchsize),:);
            sampleNeu2 = 0 .*[ones(batchsize,train_size) zeros(batchsize, dim-train_size)];
        end
        V = [(samplePos1) ; (sampleNeg1) ; (sampleNeu1)]';
        T = [(samplePos2) ; (sampleNeg2) ; (sampleNeu2)]';

        [J3, grad3_E] = gradient(V, E, D_one, T);
        J3 = mean(J3);

        %% Calculate New Transformation Matrix
        E_new = E + (learningRate * ((a * grad1_E) - (b * grad2_E) - (c * grad3_E)));
        E_ort = poldec(E_new);
        %D_one = D_one - (learningRate * c * D_one .* grad3_D);             

        %% Collect History
        J_history(iter,1) = J1;
        J_history(iter,2) = J2;
        J_history(iter,3) = J3;
        %J_history(iter,4) = norm(E_new) - 1; % comment out to speed up
        J_history(iter,5) = learningRate;

        %% Do Next Iteration
        E = E_ort;
        learningRate = learningRate * 0.99;
    end
end
    
function [Results] = evaluate(Results, a, b, c, x, E, W, dictW, train_size, lex_size, D_train, D_one, S_pos, S_neg, S_pos_test, S_neg_test, polFB, dictFB)

    dim = size(W,2);        
    W_new = (E * W')';
    %weightString = strcat('_w', sprintf('%02d',a*100),sprintf('%02d',b*100),sprintf('%02d',c*100), '_');
    %file = strcat('/mounts/data/proj/sascha/FeatureCat/data/whn_skipp-300', weightString, int2str(train_size), 'only');  
    %writeToFile(file, 'w', W_new(:,1:train_size), dictW);
    
    %% Find Oposite
    W_op = [(-1 .* W_new(:,1:train_size)) W_new(:,train_size+1:end)] ;

    %% Get Top 30
    dictP = [];
    dictP_op = [];
    dictN = [];
    dictN_op = [];
    for t=[1]
        if size(D_one,1) ~= size(D_one,2) % with bias term
            B = (D_one * [1 zeros(1,dim);zeros(dim,1) E] * [ones(size(W,1),1) W]')';
        else % no bias
            B = (D_one * E * W')';
        end
        B = B(1:5000,1);
        
        [~,id] = sort(B,'descend');
        dictP = [dictP dictW(id(1:30))];
        [id,~] = knnsearch(W_new,W_op(id(1:30),:), 'K', 2, 'Distance', 'cosine');
        id = id(:,2);
        dictP_op = [dictP_op dictW(id)];
        
        [~,id] = sort(B,'ascend');
        dictN = [dictN dictW(id(1:30))];
        [id,~] = knnsearch(W_new,W_op(id(1:30),:), 'K', 2, 'Distance', 'cosine');
        id = id(:,2);
        dictN_op = [dictN_op dictW(id)];
    end

    %% Toy Task
    positive = mean(D_train * E * S_pos', 2)';
    negative = mean(D_train * E * S_neg', 2)';
    acc = 0;
    testsize = 0;
    S_pos_trans = (D_train * E * S_pos_test')';
    cosine_pos = pdist2(S_pos_trans,positive,'euclidean');
    cosine_neg = pdist2(S_pos_trans,negative,'euclidean');
    acc = acc + sum(cosine_pos < cosine_neg);
    testsize = testsize + length(cosine_pos < cosine_neg);   
    S_neg_trans = (D_train * E * S_neg_test')';
    cosine_pos = pdist2(S_neg_trans,positive,'euclidean');
    cosine_neg = pdist2(S_neg_trans,negative,'euclidean');
    acc = acc + sum(cosine_pos > cosine_neg);
    testsize = testsize + length(cosine_pos < cosine_neg);  
    fprintf('%4.3f (%d/%d)\n', acc/testsize, acc, testsize);
    
    %% Transform Embeddings
%     if size(D_one,1) ~= size(D_one,2) % with bias term
%         W_new = (D_one * [1 zeros(1,dim);zeros(dim,1) E] * [ones(size(W,1),1) W]')';
%     else % no bias
%         W_new = (D_one * E * W')';
%     end
    W_new = W_new(:,1);

    results = [];
    %% SemEval-2015 task 10 E trial baseline
    [T, dictT] = loadTxtFile('/mounts/data/proj/sascha/FeatureCat/semeval2015_taskE_trial.txt');
    [B0, id0] = getVectors(dictT, polFB, dictFB);
    results = [results printResults('TRIAL BASELINE', T, B0, id0)];

    %% SemEval-2015 task 10 E trial
    [B, id] = getVectors(regexprep(dictT, '#', ''), W_new, regexprep(dictW, '#', ''));
    results = [results printResults('TRIAL', T, B, id)];

    %% SemEval-2015 task 10 E trial fallback
    B(id==0,:) = B0(id==0,:) + x;
    id(id==0,:) = id0(id==0,:);
    results = [results printResults('TRIAL FALLBACK', T, B, id)];

    %% SemEval-2015 task 10 E test baseline
    [T, dictT] = loadTxtFile('/mounts/data/proj/sascha/FeatureCat/semeval2015_taskE.txt');
    [B0, id0] = getVectors(dictT, polFB, dictFB);
    results = [results printResults('TEST BASELINE', T, B0, id0)];

    %% SemEval-2015 task 10 E tes
    [B, id] = getVectors(regexprep(dictT, '#', ''), W_new, regexprep(dictW, '#', ''));
    results = [results printResults('TEST', T, B, id)];

    %% SemEval-2015 task 10 E test fallback
    B(id==0,:) = B0(id==0,:) + x;
    id(id==0,:) = id0(id==0,:);
    results = [results printResults('TEST FALLBACK', T, B, id)];

    %% Add Results
    Results = [Results; a b c x train_size lex_size (acc/testsize) results];
end
    
function [] = plotGraph(J_history, num_iters)

    %% Smooth Data
    smoothing = 10;
    J_history_smooth = zeros(num_iters/smoothing,5);
    for s=1:smoothing
        J_history_smooth = J_history_smooth + J_history(s:smoothing:end,:);
    end
    J_history = J_history_smooth;
    
    %% Plot the convergence graph 
    figure('Visible','on');
    hold on;    
    plot(1:smoothing:num_iters, (J_history(:,1) / max(J_history(:,1))), '-', 'Color', [0 0.8 1], 'LineWidth', 2);
    plot(1:smoothing:num_iters, (J_history(:,2) / max(J_history(:,2))), '-', 'Color', [1 0.4 0], 'LineWidth', 2);
    plot(1:smoothing:num_iters, (J_history(:,3) / max(J_history(:,3))), '-', 'Color', [0 0.5 0], 'LineWidth', 2);
    plot(1:smoothing:num_iters, (J_history(:,4) / max(J_history(:,4))), '-', 'Color', [0.7 0 0.7], 'LineWidth', 2);
    plot(1:smoothing:num_iters, (J_history(:,5) / max(J_history(:,5))), '-', 'Color', [0.3 0.3 0.3]);
    legend('max','min', 'exac', 'norm', 'learning rate');
    xlabel('iteration');
end

function [results] = printResults(name, T, B, id)
    fprintf('%s %d/%d\n', name, sum(id~=0), length(T));
    kendall = corr(T,B,'type','Kendall');
    kendall2 = corr(T(id~=0,:),B(id~=0,:),'type','Kendall');
    fprintf('Kendall:  %4.3f (%4.3f)\n', kendall, kendall2);
    spearman = corr(T,B,'type','Spearman');
    spearman2 = corr(T(id~=0,:),B(id~=0,:),'type','Spearman');
    fprintf('Spearman: %4.3f (%4.3f)\n', spearman, spearman2);
    results = [kendall kendall2 spearman spearman2];
end

function [] = writeToFile(file, A, dictA)

    fid = fopen(file, mode);

    for i=1:size(dictA,1)
        fprintf(fid, '%s', dictA{i});
        fprintf(fid,'\t%f',A(i,:));
        fprintf(fid,'\n');
    end

    fclose(fid);

end