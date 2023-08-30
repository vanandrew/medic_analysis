function Make_cifti_distmat(fsLRdir,ciftifile,outputname,make_uint8,xhemlarge)
%Make_cifti_distmat(fsLRdir,ciftifile,outputname,[make_uint8],[xhemlarge])
%
% Given a folder with surface files and a cifti with subcortical components, this
% function will create a point-to-point distance matrix in the cifti space and save it to a
% specified output file. This matrix is useful for many functions,
% particularly infomap.
%
% The distance matrix will use geodesic distances for the distance from each
% cortical point to each other cortical point in the same cortical
% hemisphere. Euclidean distance will be used for distances between
% hemispheres, as well as for distances from cortical to volumetric (i.e.
% subcortical) points.
%
% Inputs:
% fsLRdir - location that will be searched for midthickness .surf.gii surfaces
% ciftifile - the full path to a cifti file in the same space as the
%  desired distance matrix
% outputfile - the full path of the desired output file, excluding .mat
%  extension
% make_uint8 - optional logical argument determining whether output is
%  saved in uint8 format. Default = true
% xhemlarge - optional logical argument determing whether cross-hemisphere
%  connections are set to be large numbers (to prevent distance exclusion).
%  Default = true
%
% EMG 10/22/21

if ~exist('make_uint8','var')
    make_uint8 = true;
end
if ~exist('xhemlarge','var')
    xhemlarge = true;
end

Lsurffile = dir([fsLRdir '/*.L.midthickness.*.surf.gii']);
Lsurffile = [fsLRdir '/' Lsurffile(1).name];

system(['wb_command -surface-geodesic-distance-all-to-all ' Lsurffile ' ' outputname '_L.dconn.nii'])

Rsurffile = dir([fsLRdir '/*.R.midthickness.*.surf.gii']);
Rsurffile = [fsLRdir '/' Rsurffile(1).name];

system(['wb_command -surface-geodesic-distance-all-to-all ' Rsurffile ' ' outputname '_R.dconn.nii'])


surfcoordsL = gifti(Lsurffile);
surfcoordsL = surfcoordsL.vertices;
surfcoordsR = gifti(Rsurffile); 
surfcoordsR = surfcoordsR.vertices;

cifti = ft_read_cifti_mod(ciftifile);
cifti.data = [];
maskL = cifti.brainstructure(1:length(surfcoordsL)) == 1;
maskR = cifti.brainstructure((length(surfcoordsL)+1):(length(surfcoordsL)+length(surfcoordsR))) == 2;

surfcoordsL = surfcoordsL(logical(maskL),:);
surfcoordsR = surfcoordsR(logical(maskR),:);

subcort_coords = cifti.pos(((numel(maskL)+numel(maskR))+1):end,:);

all_coords = [surfcoordsL ; surfcoordsR ; subcort_coords];
distances = single(squareform(pdist(all_coords)));
if make_uint8
    distances = uint8(distances);
end

distancesL = ft_read_cifti_mod([outputname '_L.dconn.nii']);
distancesL = distancesL.data;
distancesL = distancesL(logical(maskL),logical(maskL));
if make_uint8
    distancesL = uint8(distancesL);
end

distancesR = ft_read_cifti_mod([outputname '_R.dconn.nii']);
distancesR = distancesR.data;
distancesR = distancesR(logical(maskR),logical(maskR));
if make_uint8
    distancesR = uint8(distancesR);
end

distances(1:length(distancesL),1:length(distancesL)) = distancesL;
distances((length(distancesL)+1) : (length(distancesL) + length(distancesR)),(length(distancesL)+1) : (length(distancesL) + length(distancesR))) = distancesR;

if xhemlarge
    distances(1:length(distancesL),(length(distancesL)+1) : (length(distancesL) + length(distancesR))) = 1000;
    distances((length(distancesL)+1) : (length(distancesL) + length(distancesR)),1:length(distancesL)) = 1000;
end


 save([outputname '.mat'],'distances','-v7.3')
 
 delete([outputname '_L.dconn.nii'])
 delete([outputname '_R.dconn.nii'])
