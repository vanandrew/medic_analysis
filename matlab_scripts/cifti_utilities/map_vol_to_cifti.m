function map_vol_to_cifti(funcvol,fsLRfolder,subcort_mask,smoothing)
%map_vol_to_cifti(funcvol,[fsLRfolder],[subcort_mask],[smoothing])

if ~exist('subcort_mask') || isempty(subcort_mask)
    subcort_mask = ['/data/cn4/laumannt/subcortical_mask/subcortical_mask_LR_333.nii'];
end

if exist('fsLRfolder') && ~isempty(fsLRfolder)
    disp('mapping data to subject-specific surface')
    subspecific = true;
else
    disp('mapping data to group surface')
    fsLRfolder = '/data/cn/data1/scripts/CIFTI_RELATED/Resources/Conte69_atlas-v2.LR.32k_fs_LR.wb/';
    subspecific = false;
end

medial_masks = {'/data/nil-bluearc/GMT/Evan/Scripts/CoEScripts/Resources/cifti_masks/L.atlasroi.32k_fs_LR.shape.gii','/data/nil-bluearc/GMT/Evan/Scripts/CoEScripts/Resources/cifti_masks/R.atlasroi.32k_fs_LR.shape.gii'};
HEMS = {'L';'R'};

if subspecific
    examplefile = ls([fsLRfolder '/Native/*.L.midthickness.native.surf.gii']);
    subname_start = strfind(examplefile,'/Native/') + length('/Native/');
    subname_stop = strfind(examplefile,'.L.midthickness.native.surf.gii') - 1;
    subject = examplefile(subname_start : subname_stop);
end

if strcmp(funcvol(end-6:end),'.nii.gz')
    outname = funcvol(1:(end-7));
elseif strcmp(funcvol(end-3:end),'.nii')
    outname = funcvol(1:(end-4));
end

for hem = 1:2
    surfname = [outname '_' HEMS{hem}];
    
    if subspecific
        
        midsurf = [fsLRfolder '/Native/' subject '.' HEMS{hem} '.midthickness.native.surf.gii'];
        midsurf_LR32k = [fsLRfolder '/fsaverage_LR32k/' subject '.' HEMS{hem} '.midthickness.32k_fs_LR.surf.gii'];
        whitesurf = [fsLRfolder '/Native/' subject '.' HEMS{hem} '.white.native.surf.gii'];
        pialsurf = [fsLRfolder '/Native/' subject '.' HEMS{hem} '.pial.native.surf.gii'];
        nativedefsphere = [fsLRfolder '/Native/' subject '.' HEMS{hem} '.sphere.reg.reg_LR.native.surf.gii'];
        outsphere = [fsLRfolder '/fsaverage_LR32k/' subject '.' HEMS{hem} '.sphere.32k_fs_LR.surf.gii'];
        
        systemcall_silent(['wb_command -volume-to-surface-mapping ' funcvol ' ' midsurf ' ' surfname '.func.gii -ribbon-constrained ' whitesurf ' ' pialsurf]);
        systemcall_silent(['wb_command -metric-resample ' surfname '.func.gii ' nativedefsphere ' ' outsphere ' ADAP_BARY_AREA ' surfname '_32k_fs_LR.func.gii -area-surfs ' midsurf ' ' midsurf_LR32k]);
        delete([surfname '.func.gii']);
        
        surfs{hem} = midsurf_LR32k;
        
    else
        
        midsurf = [fsLRfolder 'Conte69.' HEMS{hem} '.midthickness.32k_fs_LR.surf.gii'];
        whitesurf = [fsLRfolder 'Conte69.' HEMS{hem} '.white.32k_fs_LR.surf.gii'];
        pialsurf = [fsLRfolder 'Conte69.' HEMS{hem} '.pial.32k_fs_LR.surf.gii'];
        
        systemcall_silent(['wb_command -volume-to-surface-mapping ' funcvol ' ' midsurf ' ' surfname '_32k_fs_LR.func.gii -ribbon-constrained ' whitesurf ' ' pialsurf]);
        
        surfs{hem} = midsurf;
        
    end
    
    surfname_final{hem} = [surfname '_32k_fs_LR.func.gii'];
    %systemcall_silent(['/usr/local/caret/bin_linux64/caret_command -file-convert -format-convert XML_BASE64 ' surfname_final{hem}]);
    
end

if strcmp(subcort_mask,'none')
    system(['wb_command -cifti-create-dense-timeseries ' outname '.dtseries.nii -left-metric ' surfname_final{1} ' -roi-left ' medial_masks{1} ' -right-metric ' surfname_final{2} ' -roi-right ' medial_masks{2}]);
else
    system(['wb_command -cifti-create-dense-timeseries ' outname '.dtseries.nii -volume ' funcvol ' ' subcort_mask ' -left-metric ' surfname_final{1} ' -roi-left ' medial_masks{1} ' -right-metric ' surfname_final{2} ' -roi-right ' medial_masks{2}]);
end
delete(surfname_final{1})
delete(surfname_final{2})


if exist('smoothing') && (smoothing > 0)
    
    smoothname = num2str(smoothing);
    smoothname(strfind(smoothname,'.')) = [];
    
    if strcmp(subcort_mask,'none')
        system(['wb_command -cifti-smoothing ' outname '.dtseries.nii ' num2str(smoothing) ' ' num2str(smoothing) ' COLUMN ' outname '_smooth' smoothname '.dtseries.nii -left-surface ' surfs{1} ' -right-surface ' surfs{2}])
    else
        system(['wb_command -cifti-smoothing ' outname '.dtseries.nii ' num2str(smoothing) ' ' num2str(smoothing) ' COLUMN ' outname '_smooth' smoothname '.dtseries.nii -left-surface ' surfs{1} ' -right-surface ' surfs{2} ' -merged-volume'])
    end
end
    