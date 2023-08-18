clc;
clear;
close all;

addpath(genpath('./read_write_cifti_32ksurface'));
addpath(genpath('./plotSpread'));
subjectsfolder = '/data/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2/';

subjects = dir([subjectsfolder '/sub-*']);

ncortverts = 59412;

group_fc = ft_read_cifti_mod('/data/nil-bluearc/GMT/Scott/ABCD/ABCD_4.5k_all.dconn.nii');
group_fc.data = group_fc.data(1:ncortverts,1:ncortverts);

similarities = [];
all_rms = [];
pct_retained = [];
subID = [];
medic_topup = [];

boldruns = cell(0,1);

similarity_maps = zeros(ncortverts,length(subjects) .* 50);

for subnum = 1:length(subjects)
    
    subject = subjects(subnum).name(5:end);
    
    sessions = dir([subjectsfolder '/sub-' subject '/ses*']);    
    
    for s = 1:length(sessions)
        sessiondir = [subjectsfolder '/sub-' subject '/' sessions(s).name];
        bolds = dir([sessiondir '/bold*']);
        
        for b = 1:length(bolds)
            
            boldnum = bolds(b).name(5:end);
            
            tmask = load([sessiondir '/bold' boldnum '/sub-' subject '_b' boldnum '_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm_tmask.txt']);
            motiondat = textread([sessiondir '/bold' boldnum '/sub-' subject '_b' boldnum '_xr3d.dat'],'%s');
            rms = str2num(motiondat{end});
            
            boldruns(end+1) = {[sessiondir '/cifti_timeseries_normalwall_atlas_freesurf/sub-' subject '_b' boldnum '_MNI152_T1_2mm_Swgt_norm_bpss_resid_LR_surf_subcort_32k_fsLR_brainstem_surfsmooth1.7_subcortsmooth1.7.dtseries.nii']};
            data = ft_read_cifti_mod([sessiondir '/cifti_timeseries_normalwall_atlas_freesurf/sub-' subject '_b' boldnum '_MNI152_T1_2mm_Swgt_norm_bpss_resid_LR_surf_subcort_32k_fsLR_brainstem_surfsmooth1.7_subcortsmooth1.7.dtseries.nii']);
            
            corr = FisherTransform(paircorr_mod(data.data(1:ncortverts,logical(tmask))'));
            corr(isnan(corr)) = 0;
            
            this_similarity = paircorr_mod_diag(group_fc.data,corr);
            this_pct_retained = nnz(tmask) ./ numel(tmask);
            
            disp([sessiondir ' ' num2str(this_pct_retained) ' ' num2str(rms) ' ' num2str(nanmean(this_similarity))])
            
            all_rms(end+1) = rms;
            subID(end+1) = subnum;
            pct_retained(end+1) = this_pct_retained;
            similarities(end+1) = nanmean(this_similarity);
            
            this_similarity(isnan(this_similarity)) = 0;
            similarity_maps(:,length(similarities)) = this_similarity;
            
            if strcmp(sessions(s).name(end-5:end),'wTOPUP')
                
                medic_topup(end+1) = 2;
                
            else
                
                medic_topup(end+1) = 1;
                
            end
            
        end
    end
    
end

similarity_maps = similarity_maps(:,1:length(similarities));

out = data;
out.data = zeros(size(out.data,1),length(similarities));
out.data(1:ncortverts,:) = similarity_maps;
out.dimord = 'pos_scalar';
out.mapname = boldruns;
ft_write_cifti_mod('Similarity_toABCDavg_MEDICandTOPUP_allruns',out);

statsout = out;
statsout.data = zeros(size(out.data,1),5);
statsout.data(1:ncortverts,1) = mean(similarity_maps(:,medic_topup==1),2);
statsout.data(1:ncortverts,2) = mean(similarity_maps(:,medic_topup==2),2);
statsout.data(1:ncortverts,3) = mean((similarity_maps(:,medic_topup==1) - similarity_maps(:,medic_topup==2)),2);
[H,P,CI,STATS] = ttest(similarity_maps(:,medic_topup==1)',similarity_maps(:,medic_topup==2)');
statsout.data(1:ncortverts,4) = STATS.tstat;
statsout.data(1:ncortverts,5) = P;
statsout.mapname = {'MEDIC Mean','TOPUP Mean','MEDIC > TOPUP Mean','MEDIC > TOPUP Paired T','MEDIC > TOPUP pval'};
ft_write_cifti_mod('Similarity_toABCDavg_MEDICvTOPUP_stats',statsout);

[H,P,CI,STATS] = ttest(similarities(medic_topup==1),similarities(medic_topup==2));
disp(['MEDIC vs TOPUP: T(' num2str(STATS.df) ')=' num2str(STATS.tstat) '; P=' num2str(P)])

%%
colors = [rand(length(subjects),1) rand(length(subjects),1) rand(length(subjects),1)];

figure;
set(gcf,'Position',[107 197 777 805])
set(gcf,'Color',[1 1 1]);%set(gcf,'Color',[.9 .9 .9])
set(gca,'Color',[1 1 1]);
hold on

plotSpread(similarities,'distributionIdx',medic_topup,'categoryIdx',subID,'categoryColors',colors)

xvals_yvals_colors_left = [];
xvals_yvals_colors_right = [];

thisax = gca;

for c = 1:length(thisax.Children)
    for p = 1:length(thisax.Children(c).XData)
        if thisax.Children(c).XData(p) < 1.5
            xvals_yvals_colors_left = [xvals_yvals_colors_left ; [thisax.Children(c).XData(p) thisax.Children(c).YData(p) thisax.Children(c).Color]];
        else
            xvals_yvals_colors_right = [xvals_yvals_colors_right ; [thisax.Children(c).XData(p) thisax.Children(c).YData(p) thisax.Children(c).Color]];
        end
    end
end

for l = 1:size(xvals_yvals_colors_left,1)
    plot([xvals_yvals_colors_left(l,1) xvals_yvals_colors_right(l,1)],[xvals_yvals_colors_left(l,2) xvals_yvals_colors_right(l,2)],'-','Color',xvals_yvals_colors_left(l,3:end))
end
for l = 1:size(xvals_yvals_colors_left,1)
    plot([xvals_yvals_colors_left(l,1) xvals_yvals_colors_right(l,1)],[xvals_yvals_colors_left(l,2) xvals_yvals_colors_right(l,2)],'.','Color',xvals_yvals_colors_left(l,3:end),'MarkerSize',40)
end

set(gca,'FontSize',30);
set(gca,'XTickLabel',{'MEDIC','TOPUP'})

%%
% 
% %%
% subjectsfolder = '/data/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline2/';
% 
% subject = '20036';
% 
% sessions = dir([subjectsfolder '/sub-' subject '/ses*']);
% 
% 
% 
% alldata = [];
% alldata_topup = [];
% 
% for s = 1:length(sessions)
%     sessiondir = [subjectsfolder '/sub-' subject '/' sessions(s).name];
%     bolds = dir([sessiondir '/bold*']);
%     
%     for b = 1:length(bolds)
%         
%         boldnum = bolds(b).name(5:end);
%         
%         disp([sessiondir '/cifti_timeseries_normalwall_atlas_freesurf/sub-' subject '_b' boldnum '_MNI152_T1_2mm_Swgt_norm_bpss_resid_LR_surf_subcort_32k_fsLR_brainstem_surfsmooth1.7_subcortsmooth1.7.dtseries.nii'])
%         
%         data = ft_read_cifti_mod([sessiondir '/cifti_timeseries_normalwall_atlas_freesurf/sub-' subject '_b' boldnum '_MNI152_T1_2mm_Swgt_norm_bpss_resid_LR_surf_subcort_32k_fsLR_brainstem_surfsmooth1.7_subcortsmooth1.7.dtseries.nii']);
%         tmask = load([sessiondir '/bold' boldnum '/sub-' subject '_b' boldnum '_faln_xr3d_uwrp_on_MNI152_T1_2mm_Swgt_norm_tmask.txt']);
%         
%         if strcmp(sessions(s).name(end-5:end),'wTOPUP')
%             
%             alldata_topup = [alldata_topup data.data(:,logical(tmask))];
%         else
%             alldata = [alldata data.data(:,logical(tmask))];
%         end
%         
%         
%     end
% end
% 
% data.data = alldata;%FisherTransform(paircorr_mod(alldata'));
% %data.dimord = 'pos_pos';
% ft_write_cifti_mod(['sub-' subject '_concat_corr_MEDIC'],data)
% data.data = alldata_topup;%FisherTransform(paircorr_mod(alldata_topup'));
% %data.dimord = 'pos_pos';
% ft_write_cifti_mod(['sub-' subject '_concat_corr_TOPUP'],data)
% 
%%

% Make table of data
subject_labels = cell(1, 370);
session_labels = cell(1, 370);
run_labels = cell(1, 370);
for i=1:length(boldruns)
    % subject label
    split_string = split(boldruns{i}, "sub-");
    split_string = split(split_string{2}, "/");
    subject_labels{i} = split_string{1};

    % session label
    if medic_topup(i) == 1
        split_string = split(boldruns{i}, "ses-");
        split_string = split(split_string{2}, "/");
        session_labels{i} = split_string{1};
    else
        split_string = split(boldruns{i}, "ses-");
        split_string = split(split_string{2}, "w");
        session_labels{i} = split_string{1};
    end

    % run label
    split_string = split(boldruns{i}, "_b");
    split_string = split(split_string{2}, "_");
    run_labels{i} = split_string{1};
end

% get medic indices
medic_indices = (medic_topup == 1);
topup_indices = (medic_topup == 2);

% get only unique subject, session, run label pairs
subject_labels = subject_labels(medic_indices);
session_labels = session_labels(medic_indices);
run_labels = run_labels(medic_indices);

% get medic similarities
medic_similarities = num2cell(real(similarities(medic_indices)));
topup_similarities = num2cell(real(similarities(topup_indices)));

% concatenate cells
datacell = [subject_labels', session_labels', run_labels', medic_similarities', topup_similarities'];

% make into datatable
datatable = cell2table(datacell, 'VariableNames', {'Subject', 'Session', 'Run', 'MEDIC', 'TOPUP'});

% save datatable to json
fid = fopen('../data/paircorr.json', 'w');
fprintf(fid, '%s', jsonencode(datatable));
fclose(fid);

