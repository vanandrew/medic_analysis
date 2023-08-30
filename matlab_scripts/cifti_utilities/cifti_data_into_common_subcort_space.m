function out = cifti_data_into_common_subcort_space(file_list)
%out = cifti_data_into_common_subcort_space(file_list)
%
%A hack that allows comparison of subcortical cifti data even when those
%data don't have the same subcortical mask. Data within the cifti files in
%the cell array 'file_list' are matched based on actual 3D coordinates, and
%put into the subcorticalspace of the first file in the list.
%
%E. Gordon 2019

col_counter = 0;

for f = 1:length(file_list)
    data = ft_read_cifti_mod(file_list{f});
    
    if f==1
        
        out = data;
        ncortverts = nnz(out.brainstructure==1) + nnz(out.brainstructure==2);
        subcort_starting_ind = max(find(out.brainstructure==2)) + 1;
        
        subcort_coords = out.pos(subcort_starting_ind:end,:);
    
    end
    
    col_inds = (col_counter+1) : (col_counter+size(data.data,2));
    
    out.data(:,col_inds) = ones(size(out.data,1),size(data.data,2)) .* NaN;
        
    out.data(1:59412,col_inds) = data.data(1:59412,:);
    
    for vox = 1:length(subcort_coords)
        D = pdist2(subcort_coords(vox,:),data.pos(subcort_starting_ind:end,:));
        zeroind = find(D==0);
        if ~isempty(zeroind)
            out.data(ncortverts+vox,col_inds) = data.data(ncortverts+zeroind,:);
        end
    end
    
    col_counter = size(out.data,2);
    
end