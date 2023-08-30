function Cluster_correct_ttestpaired_cifti(data1,data2,uncorr_pval_threshs,alphas,outputstem,iterations)
% Cluster_correct_ttest2_cifti(data1,data2,uncorr_pval_threshs,alphas,outputstem,[iterations])
%
% Conducts a paired t-test between the data in the matrix data1 and the
% matrix data2. Corrects for multiple comparisons at the cluster level by
% randomly permuting the condition identities many times, conducting
% paired t-tests, and assessing how often clusters of a given size /
% t-statistic threshold emerge in the random data.
%
% Inputs:
% data1: a cifti structure with a [#vertices X #subjects] .data field, or a path to that cifti file.
% data2: a cifti structure with a [#vertices X #subjects] .data field, or a path to that cifti file.
% height_threshs : a vector of uncorrected p thresholds to use.
% alphas: a vector containing desired alphas to correct to.
% outputstem: the base name for the cluster-corrected output t-image
% iterations: an optional input specifying the number of bootstrap iterations. Default = 1000.
%
% E.Gordon 7/2023

if ~exist('iterations','var')
    iterations = 1000;
end

if ischar(data1)
    data1 = ft_read_cifti_mod(data1);
end
if ischar(data2)
    data2 = ft_read_cifti_mod(data2);
end

temp = data1; temp.data = [];

[~,truepvals,~,trueSTATS] = ttest(data1.data',data2.data');
outcifti = data1;
outcifti.dimord = 'pos_scalar';
outcifti.data = [trueSTATS.tstat' truepvals'];
outcifti.mapname = {'T-statistic','p value'};
ft_write_cifti_mod([outputstem '_pairedT_uncorrected'],outcifti);


outcifti.mapname = cell(1,0);

concatenated = [data1.data data2.data];

groups = [ones(size(data1.data,2),1) ; ones(size(data2.data,2),1)*2];

max_random_clustersizes = zeros(iterations,1);

output_matrix = zeros(size(data1.data,1),0);

if isempty(gcp('nocreate'))
    pool = parpool(12,'IdleTimeout',30);
end

for p_thresh = uncorr_pval_threshs
    parfor iternum = 1:iterations
        
        string = ['Conducting permuted t-tests with an uncorrected p-threshold of ' num2str(p_thresh) ': iteration ' num2str(iternum) ' of ' num2str(iterations)];
        disp(string)
        
        permuted_groups = groups(randperm(length(groups)));
        [~,iterpvals,~,~] = ttest(concatenated(:,permuted_groups==1)',concatenated(:,permuted_groups==2)');
        
        input = temp;
        input.data = 1-iterpvals';
        permuted_clusteredPs = cifti_cluster(input,1-p_thresh,inf,0);
        
        permuted_clustersizes = zeros(size(permuted_clusteredPs,2),1);
        for i = 1:size(permuted_clusteredPs,2)
            permuted_clustersizes(i) = sum(permuted_clusteredPs(:,i),1);
        end
        if ~isempty(permuted_clustersizes)
            max_random_clustersizes(iternum) = max(permuted_clustersizes);
        end
    end
    disp('')
    
    sorted_max_random_clustersizes = sort(max_random_clustersizes,'descend');
    
        
    for alpha = alphas
        
        clustersize_thresh = sorted_max_random_clustersizes(round(iterations * alpha));
        input = temp;
        input.data = 1-truepvals';
        clusteredPs = cifti_cluster(input,1-p_thresh,inf,clustersize_thresh);
        output = trueSTATS.tstat' .* sum(clusteredPs,2);
        
        output_matrix(:,end+1) = output;
        outcifti.mapname(1,end+1) = {['T values for uncorrected p < ' num2str(p_thresh) ', alpha = ' num2str(alpha) ': k >= ' num2str(clustersize_thresh) ' vertices']};
        
    end
    
    outcifti.data = output_matrix;
    ft_write_cifti_mod([outputstem '_pairedT_corrected'],outcifti);
    
end


  
    
