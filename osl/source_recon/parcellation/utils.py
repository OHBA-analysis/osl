import os
import os.path as op

if False:
    # example code

    import os
    import os.path as op
    from osl.source_recon import parcellation

    workingdir = '/Users/woolrich/oslpy/osl/source_recon/parcellation/files/'
    parc_name = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm'
    tmpdir = op.join(workingdir, 'tmp')
    #os.mkdir(tmpdir)
    parcel3d_fname = op.join(workingdir, parc_name + '.nii.gz')
    parcel4d_fname = op.join(workingdir, parc_name + '_4d.nii.gz')
    parcellation.utils.convert_3dparc_to_4d(parcel3d_fname, parcel4d_fname, tmpdir, 100)

    mni_file = '/Users/woolrich/oslpy/osl/source_recon/parcellation/files/MNI152_T1_8mm_brain.nii.gz'
    spatial_res = 8 # mm
    parcel4d_ds_fname = op.join(workingdir, parc_name + '_4d_ds' + str(spatial_res) + '.nii.gz')
    parcellation.utils.spatially_downsample_nii(parcel4d_fname, parcel4d_ds_fname, mni_file, spatial_res)

def convert_3dparc_to_4d(parcel3d_fname, parcel4d_fname, tmpdir, n_parcels):
    '''

    Parameters
    ----------
    parcel3d_fname : str
        3D nifii volume with each voxel with a value of 0 if not in a parcel, or 1...p...n_parcels if in parcel p
    parcel4d_fname : str
        4D nifii output file, where each volume is a parcel
    tmpdir : str
        temp dir to write to. Must exist.
    n_parcels
        Number of parcels

    '''
    os.system('fslmaths {} {}'.format(parcel3d_fname,  op.join(tmpdir, 'parc3d.nii.gz')))
    os.system('rm -f {}'.format(parcel4d_fname))

    vol_list_str = ' '
    for pp in range(n_parcels):
        print(pp)
        vol_fname = op.join(tmpdir, 'parc3d_vol' + str(pp) + '.nii.gz')
        os.system('fslmaths {} -thr {} -uthr {} -min 1 {}'.format(parcel3d_fname, pp+0.5, pp+1.5, vol_fname))
        vol_list_str = vol_list_str + '{} '.format(vol_fname)

    os.system('fslmerge -t {} {}'.format(parcel4d_fname, vol_list_str))

def spatially_downsample_nii(file_in, file_out, file_ref, spatial_res):
    '''
    Downsample niftii file file_in spatially and writes it to file_out

    Parameters
    ----------
    file_in: str
    file_out: str
    file_ref: str
        reference niftii volume at resolution spatial_res
    spatial_res
        new spatial res in mm

    Returns
    -------
    file_out : str

    '''

    os.system('flirt -in {} -ref {} -out {} -applyisoxfm {}'.format(file_in, file_ref, file_out, spatial_res))

    return file_out

