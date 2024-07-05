################################################################################
# Calcualte contribution scores for every base and every sequence in the MPRA
#
# We use the DeepSHAP package (https://github.com/castillohair/shap/tree/castillohair/genomics_mod;
# commit f77513e2e05eb63f4d3b17ec9f9d3569c930ad02). This version contains the
# modifications introduced in the Kundaje lab to work with genomic sequences, but
# maintains compatibility with tensorflow 2. We mostly follow the instructions in
# https://docs.google.com/presentation/d/1JCLMTW7ppA3Oaz9YA2ldDgx8ItW9XHASXM1B3regxPw
# and the example in
# https://github.com/AvantiShri/colab_notebooks/blob/ecf2909528ff47dbc36c87666b44520f07cbaab2/labmeeting/Oct18/DeepSHAP_Unimodal_Input.ipynb.
# A custom score projection function (taken from the notebook above) is used to
# project the SHAP scores to the appropriate bases while calculating hypothetical
# scores for the bases that are not present. Therefore, the per-position
# contribution scores can be obtained by multiplying scores*one_hot and summing
# across the nucleotide dimension. Hypothetical scores can be used later with
# TfModisco. We used tensorflow 2 and python 3.10 to run DeepSHAP.
#
# As a reference, we use random sequences of the same length as a given input
# sequence to interpret, with the same dinucleotide distribution as the MPRA.
# References are generated previously using
# construct_reference_dinucleotide.ipynb and saved to random_seqs.pickle.
# 
# The model used is the ensemble CNN model (average output across all 10
# bootstrapped models).
# 
# Contribution scores are saved to `contributions_ensemble_cnn_model.pickle`,
# with keys corresponding to sequence IDs and values containing the contribution
# scores, as a numpy array with dimensions (# model outputs, sequence length, 4).
#
################################################################################

import json
import os
import pickle
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
timepoint_list = [2, 4, 6, 10]

lib_tpm_col = 'TPM_library'
log2_lib_tpm_col = 'log2_TPM_library'

log2_mrl_cols = [f'log2_MRL_{t}hpf' for t in timepoint_list]
delta_log2_x_cols = [f'Δlog2_X_{t}hpf' for t in timepoint_list]

res_log2_mrl_cols = [f'res_log2_MRL_{t}hpf' for t in timepoint_list]
res_delta_log2_x_cols = [f'res_Δlog2_X_{t}hpf' for t in timepoint_list]

log2_x_cols = [f'log2_X_{t}hpf' for t in timepoint_list]

# contributions output file
contributions_filepath = 'contributions_ensemble_cnn_model.pickle'

# Adapted from
# https://github.com/AvantiShri/colab_notebooks/blob/ecf2909528ff47dbc36c87666b44520f07cbaab2/labmeeting/Oct18/DeepSHAP_Unimodal_Input.ipynb
# Modified to ignore the zero padding
def combine_mult_and_diffref(mult, orig_inp, bg_data):
    assert len(orig_inp)==1
    projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
    assert len(orig_inp[0].shape)==2
    # Sum of one hot encoded sequence should give us the sequence length
    # each base that is actually present should have a 1 in the one hot matrix
    # absent bases should only contain zeros
    seq_len = int(numpy.sum(orig_inp))
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
        # Only set the bases that are actually present
        # Background should also have zeros outside this region
        # so the scores should be zero
        hypothetical_input[-seq_len:, i] = 1.0
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
        '../preprocess_data/Zb_5UTR_MPRA_preprocessed.tsv.gz',
        index_col=0,
        sep='\t',
    )
    data_full

    # Maximum sequence length
    max_seq_len = int(data_full['insert_length'].max())
    
    # Preserve only rows with input tpm above threshold
    tpm_threshold = 3

    data = data_full[data_full[lib_tpm_col] > tpm_threshold]

    print(f"{len(data_full):,} total sequences, {len(data):,} retained.")

    # Convert to one hot
    x_to_interpret = seq_utils.one_hot_encode(
        data['insert_seq'].values,
        max_seq_len=max_seq_len,
        padding='right',
        mask_val=0,
    )

    # Reference: precomputed random sequences
    # Reference depends on length
    with open('random_bg_seqs.pickle', 'rb') as handle:
        ref_seqs_all = pickle.load(handle)
    ref_seqs_onehot_all = {}
    for seq_len, ref_seqs in ref_seqs_all.items():
        ref_seqs_onehot_all[seq_len] = seq_utils.one_hot_encode(
            ref_seqs[:50],
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
    with open('../cnn/chr_splits.json', 'r') as f:
        chr_splits_info = json.load(f)

    # Load models
    model_input = layers.Input(shape=(max_seq_len, 4))
    model_list = []
    model_outputs = []
    for chr_split_idx, chr_split_info in enumerate(chr_splits_info):

        # Load model
        print("Loading model...")
        model = cnn_vgg.load_model(f'../cnn/models_cnn_vgg/model_{chr_split_idx:03d}.h5')
        model_list.append(model)

        model_output = model(model_input)
        model_outputs.append(model_output)

    # Model output: average of bs model outputs
    model_avg_output = layers.Average()(model_outputs)

    model_avg = models.Model(
        model_input,
        model_avg_output,
    )

    start_time = time.time()

    print("Initializing SHAP...")
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    explainer = shap.DeepExplainer(
        model=model_avg,
        data=ref_generator,
        combine_mult_and_diffref=combine_mult_and_diffref,
    )
    
    # Run SHAP
    print("Running SHAP...")
    raw_shap_contributions = explainer.shap_values(x_to_interpret, check_additivity=False, progress_message=10)

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
