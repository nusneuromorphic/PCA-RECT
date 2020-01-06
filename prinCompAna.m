function [dimen, comp] = prinCompAna(input_data)
% PCA (principal component analysis)
% This function has one arguments and two outputs. 
% input_data: the input raw data in size of [number_of_data x feature_dimention]
% dimen: the feature dimention that keeps 95% energy of the input_data.
% comp: the percent of energy that the "dimen" dimention data acctually holds.
tic
mean_data = mean(input_data);
% centered data matrix
center_data = input_data-ones(size(input_data,1),1)*mean_data;
% calculate the covariance matrix of the sample data
S = center_data'*center_data/(size(input_data,1)-1);
% calculate eigenvectors (loadings) W, and eigenvalues of the covariance matrix
[V,D] = eig(S);
% sort eigenvectors according to associated eigenvalues
[dummy,order] = sort(diag(-D));
% V = V(:,order);
% take the eigenvectors corresponding to the p1 and p2 biggest eigenvalues
% PC = V(:,1:pcadims);
% project the high dimension data to p-dimensional data
% proj_data = PC'*input_data';

% [COEFF SCORE latent] = princomp(input_data);

%% Energy analysis
energy = 0;total = sum(sum(abs(dummy)));
for i = 1:size(D,1)
    energy = energy + abs(dummy(i,1));
    if energy/total >= 0.95
        dimen = i;
        comp = energy/total;
%         break;
    end
    plot(i, energy/total, 'x'),hold on
end
xlabel('PC dimentions');ylabel('energy proportion');
toc
end
