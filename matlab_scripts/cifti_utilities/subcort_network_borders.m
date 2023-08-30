function subcort_network_borders(ciftinetworksfile,view)
%subcort_network_borders(ciftinetworksfile,view)
%
%Draw .5mm wide borders around the edges of networks in subcortical
% structures.
%
%ciftinetworksfile - an input cifti with discrete values in subcortex
%view - the borders only look right in one orthagonal plane. Therefore, specify
% 'axial','sagittal',or 'coronal'.

upsample_to = .5;

[path,~,~] = fileparts(ciftinetworksfile);
if isempty(path)
    path = pwd;
end

dotsloc = strfind(ciftinetworksfile,'.');
outname_base = ciftinetworksfile(1:(dotsloc(end-1)-1));

identitymatfile = '/data/cn2/data28/fsl_64/fsl_5.0.6/etc/flirtsch/ident.mat';

%networksfile = '/data/cn4/evan/RestingState/Ind_variability/Old2/MSC/wta/MSC06/MSC06_fz_wta_noMTL.dtseries.nii';
%subcortmaskfile = ['/data/nil-bluearc/GMT/Laumann/MSC/MSC06/subcortical_mask_native_freesurf/subcortical_mask_LR_333.nii'];
subcortlabelsfile = ['/data/nil-bluearc/GMT/Laumann/MSC/MSC06/subcortical_mask_native_freesurf/FreeSurferSubcorticalLabelTableLut_nobrainstem_LR.txt'];

ciftidata = ft_read_cifti_mod(ciftinetworksfile);
orig_voxelsize = abs(ciftidata.transform(1));

system(['wb_command -cifti-separate ' ciftinetworksfile ' COLUMN -volume-all ' path '/Temp.nii.gz'])

upsample_file([path '/Temp.nii.gz'],[path '/Temp_upsample.nii.gz'],identitymatfile,upsample_to,orig_voxelsize);

%upsample_file(subcortmaskfile,[path '/Temp_mask_upsample.nii.gz'],identitymatfile,upsample_to,orig_voxelsize);

%system(['wb_command -volume-label-import ' path '/Temp_mask_upsample.nii.gz ' subcortlabelsfile ' ' path '/Temp_mask_upsample_labels.nii.gz -drop-unused-labels'])

voldata = load_nii([path '/Temp_upsample.nii.gz']);
out = voldata;
out.img(:) = 0;
datasize = size(voldata.img);
inds = find(voldata.img);
for i = 1:length(inds)
    [x,y,z] = ind2sub(datasize,inds(i));
    neighbors = [x y-1 z-1;...
        x-1 y z-1;...
        x y z-1;...
        x+1 y z-1;...
        x y+1 z-1;...
        x-1 y-1 z;...
        x y-1 z;...
        x+1 y-1 z;...
        x-1 y z;...
        x+1 y z;...
        x-1 y+1 z;...
        x y+1 z;...
        x+1 y+1 z;...
        x y-1 z+1;...
        x-1 y z+1;...
        x y z+1;...
        x+1 y z+1;...
        x y+1 z+1];
    
    neighbors(any(neighbors==0,2) | (neighbors(:,1) > datasize(1)) | (neighbors(:,2) > datasize(2)) | (neighbors(:,3) > datasize(3)),:) = [];
    
    if exist('view') && ~isempty(view)
        switch view
            case 'sagittal'
                neighbors = neighbors(neighbors(:,1)==x,:);
            case 'coronal'
                neighbors = neighbors(neighbors(:,2)==y,:);
            case 'axial'
                neighbors = neighbors(neighbors(:,3)==z,:);
            
        end
        
    end
    
    neighinds = sub2ind(datasize,neighbors(:,1),neighbors(:,2),neighbors(:,3));
    
    neighvals = voldata.img(neighinds);
    
    if any(neighvals ~= voldata.img(inds(i)))
        out.img(inds(i)) = voldata.img(inds(i));
    end
    
end

if exist('view') && ~isempty(view)
    outname_base = [outname_base '_' view 'view'];
end

save_nii(out,[path '/' outname_base '_borders.nii.gz']);


% save_nii(out,[path '/Temp_upsample_borders.nii.gz']);
% 
% 
% 
% system(['wb_command -cifti-create-dense-timeseries ' outname_base '_borders.dtseries.nii -volume ' path '/Temp_upsample_borders.nii.gz ' path '/Temp_mask_upsample_labels.nii.gz']);
system(['rm ' path '/Temp*.nii.gz'])
%set_cifti_powercolors([outname_base '_borders.dtseries.nii']);

end

function upsample_file(infile,outfile,identitymatfile,upsample_to,orig_voxelsize)

upsample_shift = orig_voxelsize - upsample_to - .5;

system(['flirt -in ' infile ' -applyisoxfm ' num2str(upsample_to) ' -init ' identitymatfile ' -interp nearestneighbour -ref ' infile ' -out ' outfile])
nifti = load_nii(outfile);
nifti.hdr.hist.qoffset_x = nifti.hdr.hist.qoffset_x + (upsample_shift/2);
nifti.hdr.hist.qoffset_y = nifti.hdr.hist.qoffset_y - (upsample_shift/2);
nifti.hdr.hist.qoffset_z = nifti.hdr.hist.qoffset_z - (upsample_shift/2);
nifti.hdr.hist.srow_x = nifti.hdr.hist.srow_x + (upsample_shift/2);
nifti.hdr.hist.srow_y = nifti.hdr.hist.srow_y - (upsample_shift/2);
nifti.hdr.hist.srow_z = nifti.hdr.hist.srow_z - (upsample_shift/2);
nifti.hdr.hist.originator = nifti.hdr.hist.originator - [-(upsample_shift/2) (upsample_shift/2) (upsample_shift/2) 0 0];

save_nii(nifti,outfile);

end