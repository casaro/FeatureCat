function [A, dictA] = loadTxtFile( filename )

	fprintf('Reading word vectors ... ');
    
    fileID = fopen(filename);
    line = fgetl(fileID);
    dim = length(strfind(line,' '));
    
    frewind(fileID);
    
    textformat = ['%s', repmat(' %f',1,dim)];
    Table = textscan(fileID,textformat);
    dictA = Table{1,1}(:, 1);
    A = zeros(length(dictA),dim);
    for d=1:dim
        A(:,d) = table2array(Table(:, d+1));
    end
    
    fclose(fileID);

	fprintf('done!\n');

end