""" Utilities for pre-processing steps"""

import numpy as np
import os

import pandas
import h5py

from skimage.restoration import inpaint as sk_inpaint
from scipy.ndimage import median_filter
from scipy import special
from skimage.transform import downscale_local_mean, resize_local_mean
from skimage import filters
from sklearn.utils import shuffle

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from ulmo import defs as ulmo_defs
from ulmo.preproc import io as pp_io
from ulmo import io as ulmo_io

from IPython import embed


def build_mask(dfield, qual, qual_thresh=2, lower_qual=True,
               temp_bounds=(-2,33), field='SST'):
    """
    Generate a mask based on NaN, qual, and other bounds

    Parameters
    ----------
    dfield : np.ndarray
        Full data image
    qual : np.ndarray of int
        Quality image
    qual_thresh : int, optional
        Quality threshold value  
    lower_qual : bool, optional
        If True, the qual_thresh is a lower bound, i.e. mask values above this  
        Otherwise, mask those below!
    temp_bounds : tuple
        Temperature interval considered valid
        Used for SST
    field : str, optional
        Options: SST, aph_443

    Returns
    -------
    masks : np.ndarray
        mask;  True = bad

    """
    # Mask val
    qual_maskval = 999999 if lower_qual else -999999

    dfield[np.isnan(dfield)] = np.nan
    if field == 'SST':
        if qual is None:
            qual = np.zeros_like(dfield).astype(int)
        qual[np.isnan(dfield)] = qual_maskval
    else:
        if qual is None:
            raise IOError("Need to deal with qual for color.  Just a reminder")
        # Deal with NaN
    masks = np.logical_or(np.isnan(dfield), qual==qual_maskval)

    # Quality
    # TODO -- Do this right for color
    qual_masks = np.zeros_like(masks)

    if qual is not None and qual_thresh is not None:
        if lower_qual:
            qual_masks[~masks] = (qual[~masks] > qual_thresh) 
        else:
            qual_masks[~masks] = (qual[~masks] < qual_thresh) 

    # Temperature bounds
    #
    value_masks = np.zeros_like(masks)
    if field == 'SST':
        value_masks[~masks] = (dfield[~masks] <= temp_bounds[0]) | (dfield[~masks] > temp_bounds[1])
    # Union
    masks = np.logical_or(masks, qual_masks, value_masks)

    # Return
    return masks

def prep_table_for_preproc(tbl, preproc_root, field_size=None):
    """Prep the table for pre-processing
    e.g. add a few columns

    Args:
        tbl (pandas.DataFrame): _description_
        preproc_root (_type_): _description_
        field_size (tuple, optional): Field size. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Prep Table
    for key in ['filename', 'pp_file']:
        if key not in tbl.keys():
            tbl[key] = ''
    tbl['pp_root'] = preproc_root
    if field_size is not None:
        tbl['field_size'] = field_size[0]
    tbl['pp_idx'] = -1
    tbl['pp_type'] = ulmo_defs.mtbl_dmodel['pp_type']['init']
    # 
    return tbl

def preproc_image(item:tuple, pdict:dict, use_mask=False,
                  inpainted_mask=False):
    """
    Simple wrapper for preproc_field()
    Mainly for multi-processing

    Parameters
    ----------
    item : tuple
        field, idx or field,mask,idx (use_mask=True)
    pdict : dict
        Preprocessing dict
    use_mask : bool, optional
        If True, allow for an input mask
    inpainted_mask : bool, optional
        If True, the tuple includes an inpainted_mask
        instead of a simple mask.

    Returns
    -------
    pp_field, idx, meta : np.ndarray, int, dict

    """
    # Unpack
    if use_mask:
        field, mask, idx = item
        if inpainted_mask:
            true_mask = np.isfinite(mask)
            # Fill-in inpainted values
            field[true_mask] = mask[true_mask]
            # Overwrite
            mask = true_mask
    else:
        field, idx = item
        mask = None

    # Junk field?  (e.g. LLC)
    if field is None:
        return None

    # Run
    pp_field, meta = preproc_field(field, mask, **pdict)

    # Failed?
    if pp_field is None:
        return None

    # Return
    return pp_field.astype(np.float32), idx, meta


def preproc_field(field, mask, inpaint=True, median=True, med_size=(3,1),
                  downscale=True, dscale_size=(2,2), sigmoid=False, scale=None,
                  expon=None, only_inpaint=False, gradient=False,
                  min_mean=None, de_mean=True,
                  field_size=None,
                  fixed_km=None,
                  noise=None,
                  log_scale=False, **kwargs):
    """
    Preprocess an input field image with a series of steps:
        1. Inpainting
        2. Resize based on fixed_km (LLC)
        3. Add noise
        4. Median
        5. Downscale
        6. Sigmoid
        7. Scale
        8. Remove mean
        9. Sobel
        10. Log

    Parameters
    ----------
    field : np.ndarray
    mask : np.ndarray or None
        Data mask.  True = masked
        Required for inpainting but otherwise ignored
    inpaint : bool, optional
        if True, inpaint masked values
    median : bool, optional
        If True, apply a median filter
    med_size : tuple
        Median window to apply
    downscale : bool, optional
        If True downscale the image
    dscale_size : tuple, optional
        Size to rescale by
    noise : float, optional
        If provided, add white noise with this value
    scale : float, optional
        Scale the SSTa values by this multiplicative factor
    expon : float
        Exponate the SSTa values by this exponent
    gradient : bool, optional
        If True, apply a Sobel gradient enhancing filter
    de_mean : bool, optional
        If True, subtract the mean
    min_mean : float, optional
        If provided, require the image has a mean exceeding this value
    fixed_km : float, optional
        If provided the input image is smaller than desired, so cut it down!
    **kwargs : catches extraction keywords

    Returns
    -------
    pp_field, meta_dict : np.ndarray, dict
        Pre-processed field, mean temperature

    """
    meta_dict = {}
    # Inpaint?
    if inpaint:
        if mask.dtype.name != 'uint8':
            mask = np.uint8(mask)
        field = sk_inpaint.inpaint_biharmonic(field, mask, channel_axis=None)

    if only_inpaint:
        if np.any(np.isnan(field)):
            return None, None
        else:
            return field, None

    # Capture more metadata
    srt = np.argsort(field.flatten())
    meta_dict['Tmax'] = field.flatten()[srt[-1]]
    meta_dict['Tmin'] = field.flatten()[srt[0]]
    i10 = int(0.1*field.size)
    i90 = int(0.9*field.size)
    meta_dict['T10'] = field.flatten()[srt[i10]]
    meta_dict['T90'] = field.flatten()[srt[i90]]
    meta_dict['clear_fraction'] = 1 - np.sum(mask) / mask.size

    # Resize?
    if fixed_km is not None:
        field = resize_local_mean(field, (field_size, field_size))

    # Add noise?
    if noise is not None:
        field += np.random.normal(loc=0., 
                                  scale=noise, 
                                  size=field.shape)

    # Median
    if median:
        field = median_filter(field, size=med_size)

    # Reduce to 64x64
    if downscale:
        field = downscale_local_mean(field, dscale_size)

    # Check for junk
    if np.any(np.isnan(field)):
        return None, None

    # Check mean
    mu = np.mean(field)
    meta_dict['mu'] = mu
    if min_mean is not None and mu < min_mean:
        return None, None

    # De-mean the field
    if de_mean:
        pp_field = field - mu
    else:
        pp_field = field

    # Sigmoid?
    if sigmoid:
        pp_field = special.erf(pp_field)

    # Scale?
    if scale is not None:
        pp_field *= scale

    # Exponate?
    if expon is not None:
        neg = pp_field < 0.
        pos = np.logical_not(neg)
        pp_field[pos] = pp_field[pos]**expon
        pp_field[neg] = -1 * (-1*pp_field[neg])**expon

    # Sobel Gradient?
    if gradient:
        pp_field = filters.sobel(pp_field)
        # Meta
        srt = np.argsort(pp_field.flatten())
        i10 = int(0.1*pp_field.size)
        i90 = int(0.9*pp_field.size)
        meta_dict['G10'] = pp_field.flatten()[srt[i10]]
        meta_dict['G90'] = pp_field.flatten()[srt[i90]]
        meta_dict['Gmax'] = pp_field.flatten()[srt[-1]]

    # Log?
    if log_scale:
        if not gradient:
            raise IOError("Only implemented with gradient=True so far")
        # Set 0 values to the lowest non-zero value
        zero = pp_field == 0.
        if np.any(zero):
            min_nonz = np.min(pp_field[np.logical_not(zero)])
            pp_field[zero] = min_nonz
        # Take log
        pp_field = np.log(pp_field)


    # Return
    return pp_field, meta_dict


def preproc_tbl(data_tbl:pandas.DataFrame, 
                valid_fraction:float, 
                s3_bucket:str,
                preproc_root='standard',
                debug=False, 
                extract_folder='Extract',
                preproc_folder='PreProc',
                nsub_fields=10000, 
                use_mask=True,
                remove_local=True,
                clobber=False, 
                inpainted_mask=False,
                n_cores=10):
    """ PreProcess a set of images

    Args:
        data_tbl (pandas.DataFrame): 
            Table of input images
        valid_fraction (float): 
            Proportion of the data to select for the validation set
        s3_bucket (str): 
            Name of the S3 bucket (used to head for the Compute Node)
        preproc_root (str, optional): 
            Options for preproc
        debug (bool, optional): Defaults to False.
        extract_folder (str, optional): _description_. Defaults to 'Extract'.
        preproc_folder (str, optional): _description_. Defaults to 'PreProc'.
        nsub_fields (int, optional): _description_. Defaults to 10000.
        use_mask (bool, optional): Passed to preproc_image(). Defaults to True.
        inpainted_mask (bool, optional): _description_. Defaults to False.
        n_cores (int, optional): _description_. Defaults to 10.
        clobber (bool, optional): Defaults to False.
        remove_local (bool, optional): If True, remove the local extraction file.  Defaults to True.

    Returns:
        _type_: _description_
    """

    # Preprocess options
    pdict = pp_io.load_options(preproc_root)

    # Setup for parallel
    map_fn = partial(preproc_image, pdict=pdict,
                     use_mask=use_mask,
                     inpainted_mask=inpainted_mask)

    # Prep table
    data_tbl = prep_table_for_preproc(data_tbl, preproc_root)
    
    # Folders    mai    main_tbl['mean_temperature'] = [imeta['mu'] for imeta in meta]n_tbl['mean_temperature'] = [imeta['mu'] for imeta in meta]
    if not os.path.isdir(extract_folder):
        os.mkdir(extract_folder)                                            
    if not os.path.isdir(preproc_folder):
        os.mkdir(preproc_folder)                                            

    # Unique extraction files
    uni_ex_files = np.unique(data_tbl.ex_filename)

    for ex_file in uni_ex_files:
        print("Working on Extraction file: {}".format(ex_file))

        # Download to local
        local_file = os.path.join(extract_folder, os.path.basename(ex_file))
        if not os.path.isfile(local_file) or clobber:
            ulmo_io.download_file_from_s3(local_file, ex_file)

        # Output file -- This is prone to crash
        srep = 'inpaintT' if 'inpaintT' in local_file else 'inpaint'
        local_outfile = local_file.replace(srep,
                                           'preproc_'+preproc_root).replace(
                                               extract_folder, preproc_folder)
        s3_file = os.path.join(s3_bucket, preproc_folder, 
                               os.path.basename(local_outfile))

        # Find the matches
        gd_exfile = data_tbl.ex_filename == ex_file
        ex_idx = np.where(gd_exfile)[0]

        # 
        nimages = np.sum(gd_exfile)
        nloop = nimages // nsub_fields + ((nimages % nsub_fields) > 0)

        # Write the file locally
        # Process them all, then deal with train/validation
        pp_fields, meta, img_idx = [], [], []
        if debug:
            nloop = 3
        for kk in range(nloop):
            f = h5py.File(local_file, mode='r')

            # Load the images into memory
            i0 = kk*nsub_fields
            i1 = min((kk+1)*nsub_fields, nimages)
            print('Fields: {}:{} of {}'.format(i0, i1, nimages))
            fields = f['fields'][i0:i1]
            shape =fields.shape
            if use_mask:
                if inpainted_mask:
                    masks = f['inpainted_masks'][i0:i1]
                else:
                    masks = f['masks'][i0:i1].astype(np.uint8)
            sub_idx = np.arange(i0, i1).tolist()

            # Convert to lists
            print('Making lists')
            fields = np.vsplit(fields, shape[0])
            fields = [field.reshape(shape[1:]) for field in fields]

            # These may be inpainted_masks
            if use_mask:
                masks = np.vsplit(masks, shape[0])
                masks = [mask.reshape(shape[1:]) for mask in masks]

                items = [item for item in zip(fields,masks,sub_idx)]
            else:
                items = [item for item in zip(fields,sub_idx)]

            print('Process time')
            # Do it
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
                answers = list(tqdm(executor.map(map_fn, items,
                                                chunksize=chunksize), total=len(items)))

            # Deal with failures
            answers = [f for f in answers if f is not None]

            # Slurp
            pp_fields += [item[0] for item in answers]
            img_idx += [item[1] for item in answers]
            meta += [item[2] for item in answers]

            # Clean up
            del answers, fields, items
            if use_mask:
                del masks
            f.close()

        # Remove local_file
        if remove_local:
            os.remove(local_file)
            print("Removed: {}".format(local_file))

        # Write
        data_tbl = write_pp_fields(pp_fields, 
                                 meta, 
                                 data_tbl, 
                                 ex_idx, 
                                 img_idx,
                                 valid_fraction, 
                                 s3_file, 
                                 local_outfile,
                                 debug=debug)

        # Write to s3
        if not debug:
            ulmo_io.upload_file_to_s3(local_outfile, s3_file)

    print("Done with generating pre-processed files..")
    return data_tbl

def write_pp_fields(pp_fields:list, meta:list, 
                    main_tbl:pandas.DataFrame, 
                    ex_idx:np.ndarray,
                    ppf_idx:np.ndarray,
                    valid_fraction:float,
                    s3_file:str, local_file:str,
                    kin_meta:dict=None,
                    write_cutouts:bool=True,
                    debug:bool=False,
                    skip_meta=False):
    """Write a set of pre-processed cutouts to disk

    Args:
        pp_fields (list): List of preprocessed fields
        meta (list): List of meta measurements
            Should be same size as pp_fields
        main_tbl (pandas.DataFrame): Main table
        ex_idx (np.ndarray): Items in table extracted
            Should be same size and aligned to pp_fields
        ppf_idx (np.ndarray): Order of items in table extracted
        valid_fraction (float): Valid fraction (the rest is Train)
        s3_file (str, optional): 
            Name of the pp_file
        local_file (str):   
            Output filename locally
        kin_meta (dict, optional): Additional meta to include
        write_cutouts (bool, optional):
            Write cutouts to disk
        skip_meta (bool, optional):
            If True, don't fuss with meta data

    Returns:
        pandas.DataFrame: Updated main table
    """
    # Recast
    pp_fields = np.stack(pp_fields)
    pp_fields = pp_fields[:, None, :, :]  # Shaped for training
    pp_fields = pp_fields.astype(np.float32) # Recast


    print("After pre-processing, there are {} images ready for analysis".format(pp_fields.shape[0]))
    
    # Fill up
    main_tbl.loc[ex_idx, 'pp_file'] = s3_file

    # Ordered index by current order of pp_fields
    idx_idx = ex_idx[ppf_idx]

    # Mu
    if debug:
        embed(header='528 of preproc/utils')
    clms = list(main_tbl.keys())
    if not skip_meta:
        #main_tbl['mean_temperature'] = [imeta['mu'] for imeta in meta]
        #clms += ['mean_temperature']
        # Others
        all_tf = np.array([False]*len(main_tbl))
        all_tf[idx_idx] = True
        for key in ['mu', 'Tmin', 'Tmax', 'T90', 
                    'T10', 'clear_fraction']:
            ikey = 'mean_temperature' if key == 'mu' else key

            if key in meta[0].keys():
                # Create the column?
                if ikey not in main_tbl.keys():
                    mvalues = np.zeros(len(main_tbl))
                else:
                    mvalues = main_tbl[ikey].values
                # Fill
                mvalues[idx_idx] = [item[key] for item in meta]
                main_tbl[ikey] = mvalues
                # Add to clms
                if key not in clms:
                    clms += [key]

    # Kinematic meta
    if kin_meta is not None:
        for key in kin_meta[0].keys():
            main_tbl.loc[idx_idx, key] = [imeta[key] for imeta in kin_meta]
            # Add to clms
            if key not in clms:
                clms += [key]

    # Skip cutouts?
    if not write_cutouts:
        return main_tbl

    # Train/validation
    n = int(np.round((valid_fraction * pp_fields.shape[0])))
    idx = shuffle(np.arange(pp_fields.shape[0]))
    valid_idx, train_idx = idx[:n], idx[n:]

    if debug:
        embed(header='558 of preproc/utils')

    # Update table (this indexing is brutal..)
    #all_tf = np.array([False]*len(main_tbl))
    #all_tf[idx_idx[valid_idx]] = True
    #main_tbl.loc[all_tf, 'pp_idx'] = np.arange(valid_idx.size)
    for kk, ii in enumerate(idx_idx[valid_idx]):
        main_tbl.loc[ii, 'pp_idx'] = kk
    if len(train_idx) > 0:
        for kk, ii in enumerate(idx_idx[train_idx]):
            main_tbl.loc[ii, 'pp_idx'] = kk
    #main_tbl.loc[idx_idx[valid_idx], 'pp_idx'] = np.arange(valid_idx.size)
    #all_tf = np.array([False]*len(main_tbl))
    #all_tf[idx_idx[train_idx]] = True
    #main_tbl.loc[all_tf, 'pp_idx'] = np.arange(train_idx.size)
    #main_tbl.loc[idx_idx[train_idx], 'pp_idx'] = np.arange(train_idx.size)

    main_tbl.loc[idx_idx[valid_idx], 'pp_type'] = ulmo_defs.mtbl_dmodel['pp_type']['valid']
    main_tbl.loc[idx_idx[train_idx], 'pp_type'] = ulmo_defs.mtbl_dmodel['pp_type']['train']

    # ###################
    # Write to disk (avoids holding another 20Gb in memory)
    print("Writing: {}".format(local_file))
    with h5py.File(local_file, 'w') as f:
        # Validation
        f.create_dataset('valid', data=pp_fields[valid_idx].astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', data=main_tbl.iloc[valid_idx].to_numpy(dtype=str).astype('S'))
        dset.attrs['columns'] = clms
        # Train
        if valid_fraction < 1:
            f.create_dataset('train', data=pp_fields[train_idx].astype(np.float32))
            dset = f.create_dataset('train_metadata', data=main_tbl.iloc[train_idx].to_numpy(dtype=str).astype('S'))
            dset.attrs['columns'] = clms
    print("Wrote: {}".format(local_file))

    # Return
    return main_tbl

def write_extra_fields(fields:list, 
                    main_tbl:pandas.DataFrame, 
                    local_file:str):
    """Write an extra set of cutouts to disk

    Args:
        fields (list): List of preprocessed fields
        main_tbl (pandas.DataFrame): Main table
        s3_file (str): [description]
        local_file (str): [description]

    Returns:
        pandas.DataFrame: Updated main table
    """
    
    # Recast
    fields = np.stack(fields)
    fields = fields[:, None, :, :]  # Shaped for training
    fields = fields.astype(np.float32) # Recast

    print("After pre-processing, there are {} images ready for analysis".format(fields.shape[0]))

    # Need to be in sync with main cutouts
    valid = main_tbl.pp_type == ulmo_defs.mtbl_dmodel['pp_type']['valid']
    train = main_tbl.pp_type == ulmo_defs.mtbl_dmodel['pp_type']['train']
    ntrain = np.sum(train)

    # Prep -- this stuff is confusing!!
    sub_tbl = main_tbl[valid | train].copy()
    sub_idx = np.arange(len(sub_tbl))
    assert len(sub_idx) == fields.shape[0]

    # Fuss with indexing
    valid = sub_tbl.pp_type == ulmo_defs.mtbl_dmodel['pp_type']['valid']
    valid_idx = sub_idx[np.argsort(sub_tbl.pp_idx.values[valid])]

    # Train
    if ntrain > 0:
        train = sub_tbl.pp_type == ulmo_defs.mtbl_dmodel['pp_type']['train']
        train_idx = sub_idx[np.argsort(sub_tbl.pp_idx.values[train])]
    
    clms = list(main_tbl.keys())
    
    # ###################
    # Write to disk (avoids holding another 20Gb in memory)
    print("Writing: {}".format(local_file))
    with h5py.File(local_file, 'w') as f:
        # Validation
        f.create_dataset(
            'valid', data=fields[valid_idx].astype(np.float32))
        # Metadata
        dset = f.create_dataset('valid_metadata', 
                                data=main_tbl.iloc[valid_idx].to_numpy(dtype=str).astype('S'))
        dset.attrs['columns'] = clms
        # Train
        if ntrain > 0:
            f.create_dataset('train', data=fields[train_idx].astype(np.float32))
            dset = f.create_dataset('train_metadata', data=main_tbl.iloc[train_idx].to_numpy(dtype=str).astype('S'))
            dset.attrs['columns'] = clms
    print("Wrote: {}".format(local_file))

    # Return
    return main_tbl