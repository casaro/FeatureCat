function [] = FeatureCat(varargin)

    if 1
        wordsFilename = '/mounts/data/proj/sascha/corpora/GoogleNews-vectors-negative300.bin';
        %wordsFilename = '/mounts/data/proj/sascha/corpora/Twitter/vector_min10_300.bin';
        [W, dictW ] = loadBinaryFile(wordsFilename, 30000);
    else
        wordsFilename = '/mounts/data/proj/fadebac/sa/data_manipulated/corpora/events2012_twitter-semeval2015/mikolov/2015-01-23-train_mikolov-no_elongated-skip-50.sh/skip-50';
        wordsFilename = '/mounts/data/proj/fadebac/sa/data_manipulated/corpora/events2012_twitter-semeval2015/mikolov/2015-05-18-train_mikolov-no_elongated-skip-300.sh/skip-300';
        %wordsFilename = '/mounts/data/proj/sascha/corpora/GoogleNews-vectors-negative300_lower.txt';
        %wordsFilename = '/mounts/data/proj/sascha/corpora/GloVe/glove.twitter.27B.50d.txt';
        [W, dictW ] = loadTxtFile(wordsFilename);
        W(:,all(isnan(W),1)) = [];
    end
    resultFilename = 'results_subspace.mat';    
    
    dim = size(W,2);
    
    [U_svd S_svd V_svd] = svd(W, 'econ');
    
    dictS = [];
    polS = [];
    
    lexicons = {...
        %'/mounts/data/proj/sascha/corpora/Sentimen_Lexicon/whn_inter_gn_twitter.txt', ...
        '/mounts/data/proj/sascha/corpora/Sentimen_Lexicon/WilWieHof05.txt', ...
        %'/mounts/data/proj/sascha/corpora/Sentimen_Lexicon/HuLiu04.txt', ...
        %'/mounts/data/proj/sascha/corpora/Sentimen_Lexicon/NRC-Emotion-Lexicon.txt', ...
        %'/mounts/data/proj/sascha/corpora/Sentimen_Lexicon/NRC-Hashtag-Sentiment-Lexicon.txt', ...
        %'/mounts/data/proj/sascha/corpora/Sentimen_Lexicon/Sentiment140-Lexicon.txt', ...
        };
    
    for i=1:length(lexicons)
        fileID = fopen(lexicons{i});
        Table = textscan(fileID, '%s\t%d\n', 'CollectOutput',1);
        dictS = [dictS ; Table{1,1}(:, 1)];
        polS = [polS ; Table{1,2}(:, 1)];
        fclose(fileID);
    end
    [S, s] = getVectors(dictS, W, dictW);
    fprintf('Sentiment Lexicon training size: %d/%d\n', sum(s~=0), length(s));
    fprintf('Last word found on index %d\n', max(s));
    S(s == 0,:) = [];
    dictS(s == 0,:) = [];
    polS(s == 0,:) = [];
    s(s == 0,:) = [];
    
    % fallback lexicon
    fallback_lexicon = '/mounts/data/proj/sascha/corpora/Sentimen_Lexicon/baseline.txt';
    fileID = fopen(fallback_lexicon);
    Table = textscan(fileID, '%s\t%f\n', 'CollectOutput',1);
    dictFB = Table{1,1}(:, 1);
    polFB = Table{1,2}(:, 1);
    fclose(fileID);
    polFB = (polFB / 10) - 0.13;

    % test lexicon    
    test_lexicon = '/mounts/data/proj/sascha/corpora/Sentimen_Lexicon/HuLiu04.txt';
    fileID = fopen(test_lexicon);
    Table = textscan(fileID, '%s\t%d\n', 'CollectOutput',1);
    dictS_test = Table{1,1}(:, 1);
    polS_test = double(Table{1,2}(:, 1));
    fclose(fileID); 
    [S_test, s_test] = getVectors(dictS_test, W, dictW);
    s_test(ismember(s_test, s)) = 0; 
    fprintf('Sentiment Lexicon test size: %d/%d\n', sum(s_test~=0), length(s_test));
    S_test(s_test == 0,:) = [];
    dictS_test(s_test == 0,:) = [];
    polS_test(s_test == 0,:) = [];
    s_test(s_test == 0,:) = [];
    S_pos_test = S_test(polS_test > 0.5,:);
    S_neg_test = S_test(polS_test < -0.5,:);
        
    S_pos = S(polS > 0.9,:);
    S_neg = S(polS < -0.9,:);
	S_neu = S(polS == 0,:);
    
    num_iters = 1500;
    J_history = zeros(num_iters,5);
    batchsize = 10;
    
    Results = [];
    Weights = [];
    for a=0:0.1:1
        for b=0.0:0.1:(1-a)
            c = round((1 - a - b) * 100) / 100;
            Weights = [Weights; a b c];
        end
    end
    %Weights = [0.1 0.5 0.4];
    Weights = [0.1 0.5 0.4 length(dictS)];
    
    for w=1:size(Weights, 1)
        a = Weights(w,1);
        b = Weights(w,2);
        c = Weights(w,3);
        lex_size = Weights(w,4);

        fprintf('Weighting: %2.1f %2.1f %2.1f %3.2f\n', a, b, c, lex_size);

        for train_size=[1:5:100 110:10:300]

            E = eye(dim);
            D = eye(dim);

            s = RandStream('mt19937ar','Seed',0);
            RandStream.setGlobalStream(s);

            fprintf('Size of trained part: %d\n', train_size);
            D_train = [D(1:train_size,:);zeros(dim-train_size,size(D,2))];
            D_one = zeros(dim, dim);
            %D_one = zeros(dim, dim+1);
            D_one(1,1:train_size) = 1;
            %D_one(1,1:train_size+1) = 1;
            D_train = sparse(D_train);
            D_one = sparse(D_one);

            learningRate = 5;
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

                E_new = E + (learningRate * ((a * grad1_E) - (b * grad2_E) - (c * grad3_E)));
                E_ort = poldec(E_new);

                J_history(iter,1) = J1;
                J_history(iter,2) = J2;
                J_history(iter,3) = J3;
                %J_history(iter,4) = norm(E_new) - 1; % comment out to speed up
                J_history(iter,5) = learningRate;

                % do next iteration
                E = E_ort;
                learningRate = learningRate * 0.99;
            end

            W_new = (E * W')';
            weightString = strcat('_w', sprintf('%02d',a*100),sprintf('%02d',b*100),sprintf('%02d',c*100), '_');
            file = strcat('/mounts/data/proj/sascha/FeatureCat/data/whn_skip-300', weightString, int2str(train_size), 'only');  
            %writeToFile(file, 'w', W_new(:,1:train_size), dictW);

            % toy task
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
            % without training
            positive = mean(D_train * S_pos', 2)';
            negative = mean(D_train * S_neg', 2)';
            acc_org = 0;        
            S_pos_trans = (D_train * S_pos_test')';
            cosine_pos = pdist2(S_pos_trans,positive,'euclidean');
            cosine_neg = pdist2(S_pos_trans,negative,'euclidean');
            acc_org = acc_org + sum(cosine_pos < cosine_neg);         
            S_neg_trans = (D_train * S_neg_test')';
            cosine_pos = pdist2(S_neg_trans,positive,'euclidean');
            cosine_neg = pdist2(S_neg_trans,negative,'euclidean');
            acc_org = acc_org + sum(cosine_pos > cosine_neg);
            
            W_new = (D_one * W_new')';
            W_new = W_new(:,1);
            
            % SemEval-2015 task 10 E trial baseline
            [T, dictT] = loadTxtFile('/mounts/data/proj/sascha/FeatureCat/semeval2015_taskE_trial.txt');
            [B0, id0] = getVectors(dictT, polFB, dictFB);
            fprintf('TRIAL BASELINE: %d/%d\n', sum(id0~=0), length(T));
            k1_trial_baseline = corr(T,B0,'type','Kendall');
            k2_trial_baseline = corr(T(id0~=0,:),B0(id0~=0,:),'type','Kendall');
            fprintf('Kendall:  %4.3f (%4.3f)\n', k1_trial_baseline, k2_trial_baseline);
            s1_trial_baseline = corr(T,B0,'type','Spearman');
            s2_trial_baseline = corr(T(id0~=0,:),B0(id0~=0,:),'type','Spearman');
            fprintf('Spearman: %4.3f (%4.3f)\n', s1_trial_baseline, s2_trial_baseline);

            % SemEval-2015 task 10 E trial
            [T, dictT] = loadTxtFile('/mounts/data/proj/sascha/FeatureCat/semeval2015_taskE_trial.txt');
            [B, id] = getVectors(regexprep(dictT, '#', ''), W_new, regexprep(dictW, '#', ''));
            fprintf('TRIAL: %d/%d\n', sum(id~=0), length(T));
            k1_trial = corr(T,B,'type','Kendall');
            k2_trial = corr(T(id~=0,:),B(id~=0,:),'type','Kendall');
            fprintf('Kendall:  %4.3f (%4.3f)\n', k1_trial, k2_trial);
            s1_trial = corr(T,B,'type','Spearman');
            s2_trial = corr(T(id~=0,:),B(id~=0,:),'type','Spearman');
            fprintf('Spearman: %4.3f (%4.3f)\n', s1_trial, s2_trial);
            
            % SemEval-2015 task 10 E trial fallback
            B(id==0,:) = B0(id==0,:);
            id(id==0,:) = id0(id==0,:);
            fprintf('TRIAL FALLBACK: %d/%d\n', sum(id~=0), length(T));
            k1_trial_fallback = corr(T,B,'type','Kendall');
            k2_trial_fallback = corr(T(id~=0,:),B(id~=0,:),'type','Kendall');
            fprintf('Kendall:  %4.3f (%4.3f)\n', k1_trial_fallback, k2_trial_fallback);
            s1_trial_fallback = corr(T,B,'type','Spearman');
            s2_trial_fallback = corr(T(id~=0,:),B(id~=0,:),'type','Spearman');
            fprintf('Spearman: %4.3f (%4.3f)\n', s1_trial_fallback, s2_trial_fallback);
            
            % SemEval-2015 task 10 E baseline
            [T, dictT] = loadTxtFile('/mounts/data/proj/sascha/FeatureCat/semeval2015_taskE.txt');
            [B0, id0] = getVectors(dictT, polFB, dictFB);
            fprintf('BASELINE: %d/%d\n', sum(id0~=0), length(T));
            k1_baseline = corr(T,B0,'type','Kendall');
            k2_baseline = corr(T(id0~=0,:),B0(id0~=0,:),'type','Kendall');
            fprintf('Kendall:  %4.3f (%4.3f)\n', k1_baseline, k2_baseline);
            s1_baseline = corr(T,B0,'type','Spearman');
            s2_baseline = corr(T(id0~=0,:),B0(id0~=0,:),'type','Spearman');
            fprintf('Spearman: %4.3f (%4.3f)\n', s1_baseline, s2_baseline);

            % SemEval-2015 task 10 E
            [T, dictT] = loadTxtFile('/mounts/data/proj/sascha/FeatureCat/semeval2015_taskE.txt');
            [B, id] = getVectors(regexprep(dictT, '#', ''), W_new, regexprep(dictW, '#', ''));
            fprintf('NORMAL %d/%d\n', sum(id~=0), length(T));
            k1_normal = corr(T,B,'type','Kendall');
            k2_normal = corr(T(id~=0,:),B(id~=0,:),'type','Kendall');
            fprintf('Kendall:  %4.3f (%4.3f)\n', k1_normal, k2_normal);
            s1_normal = corr(T,B,'type','Spearman');
            s2_normal = corr(T(id~=0,:),B(id~=0,:),'type','Spearman');
            fprintf('Spearman: %4.3f (%4.3f)\n', s1_normal, s2_normal);
            
            % SemEval-2015 task 10 E fallback
            B(id==0,:) = B0(id==0,:);
            id(id==0,:) = id0(id==0,:);
            fprintf('FALLBACK %d/%d\n', sum(id~=0), length(T));
            k1_fallback = corr(T,B,'type','Kendall');
            k2_fallback = corr(T(id~=0,:),B(id~=0,:),'type','Kendall');
            fprintf('Kendall:  %4.3f (%4.3f)\n', k1_fallback, k2_fallback);
            s1_fallback = corr(T,B,'type','Spearman');
            s2_fallback = corr(T(id~=0,:),B(id~=0,:),'type','Spearman');
            fprintf('Spearman: %4.3f (%4.3f)\n', s1_fallback, s2_fallback);

            Results = [Results; a b c train_size lex_size (acc/testsize) (acc_org/testsize) ...
                k1_trial_baseline k2_trial_baseline s1_trial_baseline s2_trial_baseline ...
                k1_trial k2_trial s1_trial s2_trial ...
                k1_trial_fallback k2_trial_fallback s1_trial_fallback s2_trial_fallback ...
                k1_baseline k2_baseline s1_baseline s2_baseline ...
                k1_normal k2_normal s1_normal s2_normal ...
                k1_fallback k2_fallback s1_fallback s2_fallback];
            save(resultFilename, 'Results');
        end    
    end
    
    % Plot
    h=figure('Visible','off');
    set(gcf, 'PaperUnits', 'centimeters');
	set(gcf, 'PaperPosition', [0 0 12 5]);
	set(gcf, 'PaperSize',[12, 5]);
    plot(Results(:,5),Results(:,6),Results(:,5),Results(:,8),Results(:,5),Results(:,7));
    legend('transformed', 'svd', 'cut off', 'Location','southwest');
    ylabel('acc');
    xlabel('size of subspace');
    set(gca, 'xdir', 'reverse');
    fName = '/mounts/Users/student/sascha/paper/FeatureCat/subspace_dimensions.pdf';
    saveas(h,fName);
    close(h);
end

function [] = writeToFile(file, mode, A, dictA)

    fid = fopen(file, mode);

    for i=1:size(dictA,1)
        fprintf(fid, '%s', dictA{i});
        fprintf(fid,' %f',A(i,:));
        fprintf(fid,'\n');
    end

    fclose(fid);

end