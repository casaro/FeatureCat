function [] = writeVectors(varargin)
    
    folder = varargin{1};
    experiment = varargin{2};

    [W , dictW] = loadTxtFile(strcat(folder, 'words.txt'));

    [dictS, dictSID] = loadSynsetFile(folder);

    Theta = importdata(strcat(folder, experiment, '/theta.txt'), ' ');

    fprintf('Calculating synset vectors ... ');
    S = zeros(size(dictS, 1), size(W,2));
    for l=1:size(Theta, 1)
        w = Theta(l,1);
        s = Theta(l,2);
        theta = Theta(l, 3:end);
        S(s,:) = S(s,:) + (W(w,:) .* theta);
    end
    fprintf('done!\n');

    fprintf('Writing vectors ... ');
    file = strcat(folder, experiment, '/synsetsAndWordVectors.txt');
    fid = fopen(file, 'w');
    fprintf(fid, '%d %d\n',size(dictS, 1) + size(dictW, 1), size(W,2));
    fclose(fid);
    writeToFile(file, 'a', S, dictS);
    writeToFile(file, 'a', W, dictW);
    fprintf('done!\n');

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