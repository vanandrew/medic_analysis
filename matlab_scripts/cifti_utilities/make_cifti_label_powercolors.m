function make_cifti_label_powercolors(ciftifile)

ciftistruct = ft_read_cifti_mod(ciftifile); cifti = ciftistruct.data;
IDs = unique(cifti); IDs(IDs==0) = [];

power_surf_colormap = [1 0 0;0 0 .8;1 1 0;1 .8 .6;0 1 0;1 .6 1;0 .6 .6;0 0 0;.35 0 .65;.2 1 1;1 .5 0;.65 .25 1;0 .25 .6;.6 1 .6;.2 .3 1;.95 .95 .75;0 .4 0;.25 .25 .25];


tempfile = 'temp_forlabel.dtseries.nii';
temp = ciftistruct;
colors = zeros(length(IDs),3);

for IDnum = 1:length(IDs)
    ID = IDs(IDnum);
    
    decimalval = mod(ID,1);
    thiscolor = sum([power_surf_colormap(floor(ID),:) .* (1-decimalval) ; power_surf_colormap(ceil(ID),:) .* (decimalval)],1);
    
    colors(IDnum,:) = round(thiscolor .* 255);
    
    temp.data(ciftistruct.data==ID) = IDnum;
    
end
temp.dimord = 'pos_time';
ft_write_cifti_mod(tempfile,temp)

warning off
delete('labellist.txt');
fid = fopen('labellist.txt','at'); %open the output file for writing
fclose(fid);

for i = 1:length(IDs)
    dlmwrite('labellist.txt',['Label_' num2str(i)],'-append','delimiter','');
    dlmwrite('labellist.txt',num2str([i colors(i,:) 1]),'-append','delimiter','');
end

file_parts = tokenize(ciftifile,'.');
outname = [];
for i = 1:(length(file_parts)-2)
    outname = [outname file_parts{i} '.'];
end
outname = [outname 'dlabel.nii'];

system(['wb_command -cifti-label-import ' tempfile ' labellist.txt ' outname])
delete(tempfile)
delete('labellist.txt')
