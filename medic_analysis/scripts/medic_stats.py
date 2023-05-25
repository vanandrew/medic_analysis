from memori.pathman import PathManager as PathMan
import numpy as np
import nibabel as nib
from IPython import embed
from warpkit.utilities import corr2_coeff

# from hdf5storage import loadmat

DATA_PATH = PathMan("/home/usr/vana/GMT/David/MEDIC")

GROUP_TEMPLATE = PathMan("/home/usr/vana/GMT2/Andrew/120_Network_templates_erode3.dtseries.nii")

UPENN_DATA = PathMan("/home/usr/vana/GMT2/Andrew/UPenn/derivatives/me_pipeline/")

ASD_ADHD_DATA = PathMan("/home/usr/vana/Daenerys/ASD_ADHD/NP1173/derivatives/me_pipeline")

RESPDATA = PathMan("/home/usr/vana/GMT2/Andrew/RESPTEST/derivatives/me_pipeline")


def main():
    # import group template
    group_template = nib.load(GROUP_TEMPLATE)

    # output directory
    output_dir = PathMan("/home/usr/vana/GMT2/Andrew/MSCHD02_GROUP_CORR_ANALYSIS")
    output_dir.mkdir(exist_ok=True)

    # load dataset
    # dataset = ASD_ADHD_DATA
    # TOPUPNAME = "wTOPUP"
    # dataset = UPENN_DATA
    # TOPUPNAME = "wTOPUPSE"
    dataset = RESPDATA
    TOPUPNAME = "wTOPUP"

    # loop through subjects
    for subject in dataset.glob("sub-*"):
        print(f"Processing {subject.name}")
        for session in subject.glob("ses-*"):
            if TOPUPNAME in session.name:
                continue
            print(f"Processing {session.name}")
            group_corr_analysis(group_template, output_dir, session, subject.name, session.name, TOPUPNAME)


def group_corr_analysis(group_template, output_dir, session_path, subject, session, TOPUPNAME):
    # loop through cifti files in session
    for medic_cifti_file in session_path.glob("cifti_correlation/*.dconn.nii"):
        # get run number
        run_num = medic_cifti_file.name.split("_b")[1].split("_")[0]
        # form pepolar name
        pepolar_cifti_file = PathMan(session_path.path + TOPUPNAME) / "cifti_correlation" / medic_cifti_file.name
        medic_cifti = nib.load(medic_cifti_file)
        pepolar_cifti = nib.load(pepolar_cifti_file)

        # reindex data so it fits 32k template
        num_vertices = 32492
        group_template_structures = group_template.header.get_axis(1).iter_structures()
        group_template_left_cortex = next(group_template_structures)
        group_template_right_cortex = next(group_template_structures)
        group_template_left_cortex_slice = group_template_left_cortex[1]
        group_template_right_cortex_slice = group_template_right_cortex[1]
        group_template_left_cortex_vertices = group_template_left_cortex[2].vertex
        group_template_right_cortex_vertices = group_template_right_cortex[2].vertex
        medic_structures = medic_cifti.header.get_axis(1).iter_structures()
        medic_left_cortex = next(medic_structures)
        medic_right_cortex = next(medic_structures)
        medic_left_cortex_slice = medic_left_cortex[1]
        medic_right_cortex_slice = medic_right_cortex[1]
        medic_left_cortex_vertices = medic_left_cortex[2].vertex
        medic_right_cortex_vertices = medic_right_cortex[2].vertex
        pepolar_structures = pepolar_cifti.header.get_axis(1).iter_structures()
        pepolar_left_cortex = next(pepolar_structures)
        pepolar_right_cortex = next(pepolar_structures)
        pepolar_left_cortex_slice = pepolar_left_cortex[1]
        pepolar_right_cortex_slice = pepolar_right_cortex[1]
        pepolar_left_cortex_vertices = pepolar_left_cortex[2].vertex
        pepolar_right_cortex_vertices = pepolar_right_cortex[2].vertex

        # allocate memory for group data
        group_data = np.zeros(num_vertices)

        # fill in the data
        corr_medic = np.zeros((17, medic_right_cortex_slice.stop - medic_left_cortex_slice.start))
        corr_pepolar = np.zeros((17, pepolar_right_cortex_slice.stop - pepolar_left_cortex_slice.start))
        for i in range(17):
            # do left cortex
            group_data[group_template_left_cortex_vertices] = group_template.dataobj[
                i, group_template_left_cortex_slice
            ]

            # compute correlation between group network and network vertex
            for vertex in range(medic_left_cortex_slice.start, medic_left_cortex_slice.stop):
                group_data_subset = group_data[medic_left_cortex_vertices]
                medic_data = medic_cifti.dataobj[medic_left_cortex_slice, vertex]
                mask = ~np.isnan(medic_data)
                if np.all(mask) is False:
                    continue
                corr_medic[i, vertex] = np.corrcoef(group_data_subset[mask], medic_data[mask])[0, 1]
                print(f"MEDIC Network {i}, Vertex {vertex}")

            for vertex in range(pepolar_left_cortex_slice.start, pepolar_left_cortex_slice.stop):
                group_data_subset = group_data[pepolar_left_cortex_vertices]
                pepolar_data = pepolar_cifti.dataobj[pepolar_left_cortex_slice, vertex]
                mask = ~np.isnan(pepolar_data)
                if np.all(mask) is False:
                    continue
                corr_pepolar[i, vertex] = np.corrcoef(group_data_subset[mask], pepolar_data[mask])[0, 1]
                print(f"PEpolar Network {i}, Vertex {vertex}")

            # do right cortex
            group_data[group_template_right_cortex_vertices] = group_template.dataobj[
                i, group_template_right_cortex_slice
            ]

            for vertex in range(medic_right_cortex_slice.start, medic_right_cortex_slice.stop):
                group_data_subset = group_data[medic_right_cortex_vertices]
                medic_data = medic_cifti.dataobj[medic_right_cortex_slice, vertex]
                mask = ~np.isnan(medic_data)
                if np.all(mask) is False:
                    continue
                corr_medic[i, vertex] = np.corrcoef(group_data_subset[mask], medic_data[mask])[0, 1]
                print(f"MEDIC Network {i}, Vertex {vertex}")

            for vertex in range(pepolar_right_cortex_slice.start, pepolar_right_cortex_slice.stop):
                group_data_subset = group_data[pepolar_right_cortex_vertices]
                pepolar_data = pepolar_cifti.dataobj[pepolar_right_cortex_slice, vertex]
                mask = ~np.isnan(pepolar_data)
                if np.all(mask) is False:
                    continue
                corr_pepolar[i, vertex] = np.corrcoef(group_data_subset[mask], pepolar_data[mask])[0, 1]
                print(f"PEpolar Network {i}, Vertex {vertex}")

        # get wta corr maps
        wta_corr_medic = np.amax(corr_medic, axis=0)
        wta_corr_pepolar = np.amax(corr_pepolar, axis=0)

        # save wta corr maps
        series = nib.cifti2.SeriesAxis(start=0, step=1, size=1)
        medic_cortex = medic_left_cortex[2] + medic_right_cortex[2]
        medic_cifti_header = nib.cifti2.Cifti2Header.from_axes((series, medic_cortex))
        medic_wta_corr_cifti = nib.cifti2.Cifti2Image(wta_corr_medic[np.newaxis, :], header=medic_cifti_header)
        medic_wta_corr_cifti.to_filename(
            output_dir / f"{subject}_{session}_b{run_num}_medic_wta_corr_cifti.dtseries.nii"
        )
        pepolar_cortex = pepolar_left_cortex[2] + pepolar_right_cortex[2]
        pepolar_cifti_header = nib.cifti2.Cifti2Header.from_axes((series, pepolar_cortex))
        pepolar_wta_corr_cifti = nib.cifti2.Cifti2Image(wta_corr_pepolar[np.newaxis, :], header=pepolar_cifti_header)
        pepolar_wta_corr_cifti.to_filename(
            output_dir / f"{subject}_{session}_b{run_num}_pepolar_wta_corr_cifti.dtseries.nii"
        )
