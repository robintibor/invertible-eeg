import pandas as pd
from tqdm.autonotebook import tqdm
import os.path
import os


def delete_low_quality_model_params(population_csv_filename, n_best=500):
    population_df = pd.read_csv(population_csv_filename, index_col="pop_id")
    to_delete_encodings_df = population_df.sort_values(by="valid_mis").iloc[n_best:]
    delete_model_params(to_delete_encodings_df)


def delete_model_params(to_delete_encodings_df):
    for folder in tqdm(to_delete_encodings_df.folder):
        encoding_filename = os.path.join(folder, "encoding.pth")
        if os.path.isfile(encoding_filename):
            os.remove(encoding_filename)
        # from invertibleeeg.experiments.nas import copy_clean_encoding_dict
        # for encoding_filename in (os.path.join(folder, 'encoding.pth'), os.path.join(folder, 'encoding_no_params.pth')):
        #    encoding = th.load(encoding_filename, map_location='cpu')
        #    clean_encoding = copy_clean_encoding_dict(encoding)
        #    th.save(clean_encoding, encoding_filename)


def get_parent_ids(start_id, population_df):
    parent_ids = []
    this_exp = population_df.loc[start_id]
    parent_ids.append(this_exp.name)
    while isinstance(this_exp.parent_folder, str):
        this_exp = population_df[population_df.folder == this_exp.parent_folder].iloc[0]
        parent_ids.append(this_exp.name)
    return parent_ids
