function upsample_cifti_to_164(cifti_in_name,discrete)
%upsample_cifti_to_164(cifti_in_name,discrete)
%
%Upsamples a cifti in 32k_fs_LR space to 164k_fs_LR space, usually for
% display purposes.
%
%Input should be a cifti file
%
%The 'discrete' option should be set to true or false and governs whether
% the cifti data should be usampled using nearest neighbor inteprolation
% (useful if the file contains discrete values, i.e. a network file).
%
%E. Gordon 2022

if ~exist('discrete','var')
    discrete = false;
else
    discrete = logical(discrete);
end

[outpath,~,~] = fileparts(cifti_in_name);
if isempty(outpath)
    outpath = [pwd '/'];
end


dotsloc = strfind(cifti_in_name,'.');
cifti_out_name = [cifti_in_name(1:(dotsloc(end-1)-1)) '_164' cifti_in_name(dotsloc(end-1) : end)];

template_file_164 = '/data/nil-bluearc/GMT/Evan/Atlases/120_LR_minsize400_recolored_manualconsensus_LR_164.dtseries.nii';

L_sphere_32 = '/data/cn/data1/scripts/CIFTI_RELATED/Resources/Conte69_atlas-v2.LR.32k_fs_LR.wb/Conte69.L.sphere.32k_fs_LR.surf.gii';
R_sphere_32 = '/data/cn/data1/scripts/CIFTI_RELATED/Resources/Conte69_atlas-v2.LR.32k_fs_LR.wb/Conte69.R.sphere.32k_fs_LR.surf.gii';

L_sphere_164 = '/data/cn/data1/scripts/CIFTI_RELATED/Resources/Conte69_atlas.LR.164k_fs_LR//Conte69.L.sphere.164k_fs_LR.surf.gii';
R_sphere_164 = '/data/cn/data1/scripts/CIFTI_RELATED/Resources/Conte69_atlas.LR.164k_fs_LR//Conte69.R.sphere.164k_fs_LR.surf.gii';

cifti_in = ft_read_cifti_mod(cifti_in_name);

if discrete
    method = 'BARYCENTRIC -surface-largest';
else
    method = 'ADAP_BARY_AREA';
end

system(['wb_command -cifti-resample ' cifti_in_name ' COLUMN ' template_file_164 ' COLUMN ' method ' TRILINEAR ' outpath 'temp_forupsample' cifti_in_name(dotsloc(end-1) : end) ' -left-spheres ' L_sphere_32 ' ' L_sphere_164 ' -right-spheres  ' R_sphere_32 ' ' R_sphere_164]);

surf_only_data = ft_read_cifti_mod([outpath 'temp_forupsample' cifti_in_name(dotsloc(end-1) : end)]);

data_subcort = cifti_in.data(cifti_in.brainstructure(cifti_in.brainstructure>0)>2,:);
brainstructurelabels_subcort = cifti_in.brainstructurelabel(3:end);
brainstructure_subcort = cifti_in.brainstructure(cifti_in.brainstructure>2);
pos_subcort = cifti_in.pos(cifti_in.brainstructure>2,:);

surf_only_data.data((end+1) : (end+size(data_subcort,1)),:) = data_subcort;
surf_only_data.brainstructurelabel((end+1) : (end+length(brainstructurelabels_subcort))) = brainstructurelabels_subcort;
surf_only_data.brainstructure((end+1) : (end+length(brainstructure_subcort))) = brainstructure_subcort;
surf_only_data.pos((end+1) : (end+size(pos_subcort,1)),:) = pos_subcort;

ft_write_cifti_mod(cifti_out_name,surf_only_data)

delete([outpath 'temp_forupsample' cifti_in_name(dotsloc(end-1) : end)])