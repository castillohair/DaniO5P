################################################################################
# Calcualte contribution scores for every base and every sequence in the MPRA
#
# We use the DeepSHAP package (https://github.com/kundajelab/shap; commit
# 29d2ffab405619340419fc848de6b53e2ef0f00c), mostly following the instructions
# in https://docs.google.com/presentation/d/1JCLMTW7ppA3Oaz9YA2ldDgx8ItW9XHASXM1B3regxPw
# and the example in
# https://github.com/AvantiShri/colab_notebooks/blob/ecf2909528ff47dbc36c87666b44520f07cbaab2/labmeeting/Oct18/DeepSHAP_Unimodal_Input.ipynb.
# A custom score projection function (taken from the notebook above) is used to
# project the SHAP scores to the appropriate bases while calculating hypothetical
# scores for the bases that are not present. Therefore, the per-position
# contribution scores can be obtained by multiplying scores*one_hot and summing
# across the nucleotide dimension. Hypothetical scores can be used later with
# TfModisco. We used tensorflow 1.15 and python 3.6 to run DeepSHAP.
#
# As a reference, we use random sequences of the same length as a given input
# sequence to interpret, with the same dinucleotide distribution as the MPRA.
# References are generated previously using
# construct_reference_dinucleotide.ipynb and saved to random_seqs.pickle.
# 
# The model used is the ensemble CNN model (average output across all 10
# bootstrapped models). Models trained in tensorflow 2 did not load properly in
# tensorflow 1, so we had to rebuild the models from scratch and manually load
# weights from the trained models.
# 
# Contribution scores are saved to `contributions_ensemble_cnn_model.pickle`,
# with keys corresponding to sequence IDs and values containing the contribution
# scores, as a numpy array with dimensions (# model outputs, sequence length, 4).
#
################################################################################

import json
import os
import pickle
import pickle5
import sys
import time

import numpy
import numpy as np
import matplotlib
from matplotlib import pyplot
import pickle
import pandas
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models

import shap

utils_dir = '../utils'
sys.path.append(utils_dir)
import seq_utils
import cnn_vgg

# Sublibraries
tpm_fraction_list = ['input', '80S', 'LMW', 'HMW']
pol_fraction_list = ['80S', 'LMW', 'HMW']
timepoint_list = [2, 4, 6, 10]

min_input_tpm_col = 'min_TPM_input'

tpm_cols = [f'gmean_TPM_{f}_{t}hpf' for f in tpm_fraction_list for t in timepoint_list]
input_tpm_cols = [f'gmean_TPM_input_{t}hpf' for t in timepoint_list]
input_log2_tpm_cols = [f'log2_TPM_input_{t}hpf' for t in timepoint_list]
diff_log2_tpm_input_cols = [f'diff_log2_TPM_input_{t}-2hpf' for t in timepoint_list[1:]]

mrl_cols = [f'MRL_{t}hpf' for t in timepoint_list]
log2_mrl_cols = [f'log2_MRL_{t}hpf' for t in timepoint_list]

res_log2_mrl_cols = [f'res_log2_MRL_{t}hpf' for t in timepoint_list]
res_diff_log2_tpm_input_cols = [f'res_diff_log2_TPM_input_{t}-2hpf' for t in timepoint_list[1:]]



# contributions output file
contributions_filepath = 'contributions_ensemble_cnn_model.pickle'

# The following was copied from 
# https://github.com/AvantiShri/colab_notebooks/blob/ecf2909528ff47dbc36c87666b44520f07cbaab2/labmeeting/Oct18/DeepSHAP_Unimodal_Input.ipynb
def combine_mult_and_diffref(mult, orig_inp, bg_data):
    assert len(orig_inp)==1
    projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
    assert len(orig_inp[0].shape)==2
    #At each position in the input sequence, we iterate over the one-hot encoding
    # possibilities (eg: for genomic sequence, this is ACGT i.e.
    # 1000, 0100, 0010 and 0001) and compute the hypothetical 
    # difference-from-reference in each case. We then multiply the hypothetical
    # differences-from-reference with the multipliers to get the hypothetical contributions.
    #For each of the one-hot encoding possibilities,
    # the hypothetical contributions are then summed across the ACGT axis to estimate
    # the total hypothetical contribution of each position. This per-position hypothetical
    # contribution is then assigned ("projected") onto whichever base was present in the
    # hypothetical sequence.
    #The reason this is a fast estimate of what the importance scores *would* look
    # like if different bases were present in the underlying sequence is that
    # the multipliers are computed once using the original sequence, and are not
    # computed again for each hypothetical sequence.
    for i in range(orig_inp[0].shape[-1]):
        hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")
        hypothetical_input[:,i] = 1.0
        hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[0])
        hypothetical_contribs = hypothetical_difference_from_reference*mult[0]
        projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1) 
    return [np.mean(projected_hypothetical_contribs,axis=0)]

if __name__=='__main__':
    
     # Load contributions file if already exists
    if os.path.exists(contributions_filepath):
        with open(contributions_filepath, 'rb') as handle:
            contributions_dict = pickle.load(handle)
        print(f"Loaded {len(contributions_dict)} preexisting contributions.")
    else:
        print(f"Initializing contributions data structure...")
        contributions_dict = {}

    # Load data
    data_full = pandas.read_csv(
        '../00_data/Zb_5UTR_MPRA_TPM_MRL.tsv.gz',
        index_col=0,
        sep='\t',
    )
    data_full

    # Maximum sequence length
    max_seq_len = data_full['insert_length'].max()
    
    # Preserve only rows with input tpm above threshold
    tpm_threshold = 2

    data = data_full[data_full[min_input_tpm_col] > tpm_threshold]

    print(f"{len(data_full):,} total sequences, {len(data):,} retained by read counts.")

    # Convert to one hot
    x_to_interpret = seq_utils.one_hot_encode(
        data['insert_seq'].values,
        max_seq_len=max_seq_len,
        padding='right',
        mask_val=0,
    )

    # Reference: precomputed random sequences
    # Reference depends on length
    with open('random_seqs.pickle', 'rb') as handle:
        ref_seqs_all = pickle5.load(handle)
    ref_seqs_onehot_all = {}
    for seq_len, ref_seqs in ref_seqs_all.items():
        ref_seqs_onehot_all[seq_len] = seq_utils.one_hot_encode(
            ref_seqs[:25],
            max_seq_len=max_seq_len,
            padding='right',
            mask_val=0,
        )
    
    # The following function should give a different reference depending on the input's length
    def ref_generator(input_modes_list):
        assert(len(input_modes_list)==1)
        input_onehot = input_modes_list[0]
        # Sum of one hot encoded sequence should give us the sequence length
        # each base that is actually present should have a 1 in the one hot matrix
        # absent bases should only contain zeros
        seq_len = numpy.sum(input_onehot)
        ref_seqs_onehot = ref_seqs_onehot_all[seq_len]
        return [ref_seqs_onehot]
    
    # Load chromosome split
    with open('../02_cnn/chr_splits.json', 'r') as f:
        chr_splits_info = json.load(f)

    # Individual models needs to be rebuilt from scratch since we're using tensorflow 1
    # But we will load the weights from the previously trained models
    model_input = layers.Input(shape=(max_seq_len, 4))
    model_list = []
    model_outputs = []
    for chr_split_idx, chr_split_info in enumerate(chr_splits_info):

        # Load model
        print("Loading model...")
        model = cnn_vgg.make_model(
            input_seq_len=max_seq_len,
            conv_layers=3,
            conv_kernel_size=7,
            conv_activation='relu',
            conv_filters_first=128,
            conv_dropout=0.1,
            dense_units=[150],
            dense_activation=['relu'],
            dense_dropout=[0.0],
            n_outputs=len(res_log2_mrl_cols + res_diff_log2_tpm_input_cols),
        )
        model.load_weights(f'../02_cnn/models_cnn_vgg/model_{chr_split_idx:03d}.h5')

        model_list.append(model)

        model_output = model(model_input)
        model_outputs.append(model_output)

    # Model output: average of bs model outputs
    model_avg_output = layers.Average()(model_outputs)

    # Define bootstrapped model
    model_avg = models.Model(
        model_input,
        model_avg_output,
    )

    start_time = time.time()

    print("Initializing SHAP...")
    explainer = shap.DeepExplainer(
        model=model_avg,
        data=ref_generator,
        combine_mult_and_diffref=combine_mult_and_diffref,
    )
    
    # Run SHAP
    print("Running SHAP...")
    raw_shap_contributions = explainer.shap_values(x_to_interpret)

    end_time = time.time()
    print(f"Took {end_time - start_time} secs for {len(x_to_interpret)} sequences.")

    # Attempt to save contributions
    seqs_contributions_by_seq = numpy.array(raw_shap_contributions).swapaxes(0, 1)
    seqs_ids = data.index.to_list()
    assert(len(seqs_contributions_by_seq)==len(seqs_ids))

    for seq_id, seq_contribution in zip(seqs_ids, seqs_contributions_by_seq):
        if seq_id in contributions_dict:
            print(f"WARNING: contributions for seq_id {seq_id} already present. Skipping...")
            continue
        contributions_dict[seq_id] = seq_contribution

    # Save
    with open(contributions_filepath, 'wb') as handle:
        print(f"Saving...")
        pickle.dump(contributions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print()

    print("Done.")
