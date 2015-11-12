function [J_history, E] = train(weighting, E, D_sent, D_conc, D_freq, num_iters, learning_rate, batchsize, W, s_pos, s_neg, c_pos, c_neg)

    J_history = zeros(num_iters,length(weighting) + 2);
    
    for iter=1:num_iters
        
        int_batchsize = int32(batchsize);
        half_batchsize = idivide(int_batchsize, 2);
        
        E_new = E;

        %% Max diff Sentiment
        if (weighting(1) > 0)
            samplePos = W(s_pos(randsample(1:length(s_pos), int_batchsize, true)),:);
            sampleNeg = W(s_neg(randsample(1:length(s_neg), int_batchsize, true)),:);
            V = (samplePos - sampleNeg)';

            [J1, grad1_E] = gradient(V, E, D_sent, zeros(size(D_sent, 1), size(V, 2)));
            J_history(iter,1) = mean(J1);
            
            if (mod(iter,6) == 0)
                E_new = E_new + (learning_rate * weighting(1) * grad1_E);
            end
        end

        %% Min same Sentiment
        if (weighting(2) > 0)
            samplePos1 = W(s_pos(randsample(1:length(s_pos), half_batchsize, true)),:);
            sampleNeg1 = W(s_neg(randsample(1:length(s_neg), half_batchsize, true)),:);
            samplePos2 = W(s_pos(randsample(1:length(s_pos), half_batchsize, true)),:);
            sampleNeg2 = W(s_neg(randsample(1:length(s_neg), half_batchsize, true)),:);
            V = [(samplePos1 - samplePos2) ; (sampleNeg1 - sampleNeg2)]';

            [J2, grad2_E] = gradient(V, E, D_sent, zeros(size(D_sent, 1), size(V, 2)));
            J_history(iter,2) = mean(J2);
            
            if (mod(iter,6) == 1)
                E_new = E_new - (learning_rate * weighting(2) * grad2_E);
            end
        end
        
        %% Max diff Concreteness
        if (weighting(3) > 0)
            samplePos = W(c_pos(randsample(1:length(c_pos), int_batchsize, true)),:);
            sampleNeg = W(c_neg(randsample(1:length(c_neg), int_batchsize, true)),:);
            V = (samplePos - sampleNeg)';

            [J3, grad3_E] = gradient(V, E, D_conc, zeros(size(D_conc, 1), size(V, 2)));
            J_history(iter,3) = mean(J3);
            
            if (mod(iter,6) == 2)
                E_new = E_new + (learning_rate * weighting(3) * grad3_E);
            end
        end

        %% Min same Concreteness
        if (weighting(4) > 0)
            samplePos1 = W(c_pos(randsample(1:length(c_pos), half_batchsize, true)),:);
            sampleNeg1 = W(c_neg(randsample(1:length(c_neg), half_batchsize, true)),:);
            samplePos2 = W(c_pos(randsample(1:length(c_pos), half_batchsize, true)),:);
            sampleNeg2 = W(c_neg(randsample(1:length(c_neg), half_batchsize, true)),:);
            V = [(samplePos1 - samplePos2) ; (sampleNeg1 - sampleNeg2)]';

            [J4, grad4_E] = gradient(V, E, D_conc, zeros(size(D_conc, 1), size(V, 2)));
            J_history(iter,4) = mean(J4);
            
            if (mod(iter,6) == 3)
                E_new = E_new - (learning_rate * weighting(4) * grad4_E);
            end
        end
        
        %% Maximize diff Freq
        if (weighting(5) > 0)
            sampleHead = W(randsample(1:1000, int_batchsize, true),:);
            sampleTail = W(randsample(10000:11000, int_batchsize, true),:);
            V = (sampleHead - sampleTail)';

            [J5, grad5_E] = gradient(V, E, D_freq, zeros(size(D_freq, 1), size(V, 2)));
            J_history(iter,5) = mean(J5);
            
            if (mod(iter,6) == 4)
                E_new = E_new + (learning_rate * weighting(5) * grad5_E);
            end
        end
        
        %% Minimize same Freq
        if (weighting(6) > 0)
            sample_ids = randsample(1:10000, int_batchsize, true);
            sample1 = W(sample_ids,:);
            sample2 = W(sample_ids + 1,:);
            V = (sample1 - sample2)';

            [J6, grad6_E] = gradient(V, E, D_freq, zeros(size(D_freq, 1), size(V, 2)));
            J_history(iter,6) = mean(J6);
            
            if (mod(iter,6) == 5)
                E_new = E_new - (learning_rate * weighting(6) * grad6_E);
            end
        end

        %J_history(iter,7) = norm(E_new) - 1; % comment out to speed up
        J_history(iter,8) = learning_rate;

        %% Orthogonalize
        E_new = poldec(E_new);
        E = E_new;
        learning_rate = learning_rate * 0.99;
        batchsize = batchsize;% * 1.03;
    end
end