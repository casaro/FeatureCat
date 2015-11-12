function [ X , y ] = getVectors( dictX, A, dictA )

    X = zeros(size(dictX,1),size(A,2));
	y = zeros(size(dictX,1),1);
    
    for i=1:size(dictX,1)
        ind = strcmpi(dictX{i}, dictA);
        if (any(ind))
            y(i) = find(ind,1);
            X(i,:) = A(y(i),:);
            continue;
        end
        continue;
        
        C = strsplit(dictX{i},'_');
        ind = strcmpi(C{end}, dictA);
        if (any(ind))
            y(i) = find(ind,1);
            X(i,:) = -0.2 * A(y(i),:);
            continue;
        end
    end
    
end

