function [J, grad_E, grad_D] = gradient(input, E, D, output)

    if size(D,2) ~= size(E,1) % if with bias term
        dim = size(input, 1);
        batchsize = size(input, 2);
        E = [1 zeros(1,dim);zeros(dim,1) E];
        input = [ones(1,batchsize);input];
    end

	% precalculations
    m = size(output, 2);
	predict = (D * E * input);
    diff = predict - output;
    J = mean(diff.^2);
    
	d = diff' * D;

	% calculate derivate for E
	grad_E = d' * input' / m;
    
    if size(D,2) ~= size(E,1) % if with bias term
        grad_E = grad_E(2:end,2:end);
    end
    
    if nargout > 2
        % calculate derivate for D
        e = E * input;
        grad_D = diff * e' / m;
    end
end
