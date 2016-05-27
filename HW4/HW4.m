clear;clc

load('Yale_32x32.mat');

fea2 = fea - mean(fea(:));

%coeff = pca(fea2', 'Algorithm', 'svd');



Cov_mat = fea*fea'/size(fea,2);
% 
[eig_vecs, diags] = eig(Cov_mat);
% 
[sv, si] = sort(diag(diags),'descend');
% 
Vs = eig_vecs(:,si);

pca_to_save = [Vs gnd];
csvwrite('pca_components.csv', pca_to_save)

[A, E] = proximal_gradient_rpca(fea, 0.02);

Cov_mat = A*A'/size(A,2);
% 
[eig_vecs, diags] = eig(Cov_mat);
% 
[sv, si] = sort(diag(diags),'descend');
% 
PCA_RPCA = eig_vecs(:,si);

rpca_to_save = [A gnd];
csvwrite('rpca_components.csv', rpca_to_save)

pca_rpca_to_save = [PCA_RPCA gnd];
csvwrite('pca_on_rpca.csv', pca_rpca_to_save)

