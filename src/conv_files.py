import os
import gzip
import shutil

def decompress_files(data_filepath, new_filepath):
# pass in the folderpath of the raw .tcx.gz files as the 'data_filepath' variable, and location for the uncompressed files
# to be stored in as 'new_filepath'
    if not os.path.exists(new_filepath):
        os.makedirs(new_filepath)
    for item in os.listdir(data_filepath):
        if item.endswith('.tcx.gz') or item.endswith('.fit.gz'):
            with gzip.open(data_filepath + item, 'rb') as f_in:
                if item.endswith('.tcx.gz'):
                    # convert tcx to xmls
                    old = f_in.readlines()
                    # for some reason the xml files start with 10 spaces. this deletes them
                    old[0] = old[0][10:]
                    with open(new_filepath + item[:-7] + '.xml', 'wb') as f_out:
                        f_out.writelines(old)
                else:
                    # saves the decompressed fit files
                    with open(new_filepath + item[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

decompress_files('../raw_data/', '../data/')
