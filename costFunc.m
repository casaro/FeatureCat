function [J] = costFunc(input, E, D, output)

    predict = (D * E * input);
    diff = predict - output;
    J = mean(diff.^2);

end