function rotated_maps = rotate_cifti(cifti_tobe_rotated,premade_rotations_cifti,baddata_cifti,outfilename)
%rotated_maps = rotate_cifti(cifti_tobe_rotated,rotations_cifti,baddata_cifti,[outfilename])
%
%Create a set of randomly rotated versions of a single-column cifti.
%
%
% 'cifti_tobe_rotated' - the input cifti file that will be rotated.
%
% 'premade_rotations' - a set of rotations to use, created as with
% Make_rotations.m. Omit or leave empty ([]) to use the rotations in
% /data/nil-bluearc/GMT/Evan/MSC/reliability_correction/Rotated_inds.dtseries.nii.
%
% 'baddata_cifti' - a set of vertices that are assumed to be "bad" (i.e.,
% in susceptibility artifact areas) and so should be blanked out of all
% rotations.
%
% 'outfilename' - if specified, the rotated maps will be written out to
% this filename.
%
%E. Gordon 2020

if ischar(cifti_tobe_rotated)
    map_tobe_rotated = ft_read_cifti_mod(cifti_tobe_rotated);
else
    map_tobe_rotated = cifti_tobe_rotated;
end
ncortverts = nnz(map_tobe_rotated.brainstructure(map_tobe_rotated.brainstructure>0)<3);
map_tobe_rotated.data((ncortverts+1):end) = 0;

if ~exist('premade_rotations_cifti','var') || isempty(premade_rotations_cifti)
    premade_rotations_cifti = '/data/nil-bluearc/GMT/Evan/MSC/reliability_correction/Rotated_inds.dtseries.nii';
end
rotations = ft_read_cifti_mod(premade_rotations_cifti);

if exist('baddata_cifti','var') && ~isempty(baddata_cifti)
    if ischar(baddata_cifti)
        baddata = ft_read_cifti_mod(baddata_cifti);
    else
        baddata = baddata_cifti;
    end
    baddatainds = find(baddata.data); baddatainds(baddatainds>ncortverts) = [];
    map_tobe_rotated.data(baddatainds) = 0;
else
    baddatainds = [];
end
    
rotated_maps = rotations.data; rotated_maps(:) = 0;
rotated_maps(rotations.data>0) = map_tobe_rotated.data(rotations.data(rotations.data>0));
rotated_maps(baddatainds,:) = 0;

if exist('outfilename','var')
    
    out = rotations;
    out.data = rotated_maps;
    out.dimord = map_tobe_rotated.dimord;
    
    ft_write_cifti_mod(outfilename,out);
    
end

