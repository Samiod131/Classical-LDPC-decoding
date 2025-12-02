import time
import json
import warnings
import sys

import numpy as np

import tntools_ldpc_decoder as tnt_ldpc
from ldpc_rand_codegen import get_rand_code
import utils

"""
This code evaluates performances of randomly selected bandwidth reduced LDPC codes using the LDPC decoder procedure. 

One would normally replace the content of the code_select to select a specific decoder with optimized properties.
Only the all-zeros codeword is being decoded here. By properities of LDPC codes, this codeword won't have a larger distance
 to other codewords than any other statistically. 
"""


def bitflip_array(p, n):
    '''
    Return 0-array of n bits with fixed bit flip probability p
    '''
    arr = np.random.choice([0, 1], n, p=[1-p, p])
    return arr


class TimedDecoder:
    '''
    To keep time of each decoding procedure. 
    '''

    def __init__(self, decoder):
        self.decoder = decoder
        self.times = []

    def decode(self, entry):
        before = time.time()
        output = self.decoder.decode(entry)
        after = time.time()

        self.times.append(after-before)

        return output


def code_select(bit_degree, check_degree, n_mult):
    '''
    Selects a random LDPC code with reduced bandwidth.
    '''
    ldpc_code = get_rand_code(
        bit_deg=bit_degree, check_deg=check_degree, nmult=n_mult, rcmk=True)

    return ldpc_code


def new_run_func(code_params, general_params, decoder_params, svd_params, main_comp_params):
    '''
    Run one serie of experience for a set of parameters and returns the required
    parameters results in a dictionnary format.
    '''

    # Gets the right LDPC code for the testing
    ldpc_code = code_select(**code_params)

    entry_size = code_params['check_degree']*code_params['n_mult']

    

    # Default case for minimal noise cutting in svd function
    if svd_params['err_th'] == 'default':
        # minimum possible probability value
        svd_params['err_th'] = decoder_params['b_prob']**(entry_size)/10

    # Default max bond for dephased dmrg is the same as for the whole schedule
    if main_comp_params['chi_max'] == 'default':
        main_comp_params['chi_max'] = svd_params['max_len']

    decoder = TimedDecoder(tnt_ldpc.TN_LDPC_Decoder(
        parity_mat=ldpc_code, 
        decoder_par=decoder_params, 
        svd_function_par=svd_params, 
        main_comp_finder_par=main_comp_params)
        )
    
    # Create list for storing all failures / successes
    failures = []


    for _ in range(general_params['samples']):
        # Generate Random entry with given error rate
        entry = bitflip_array(p=general_params['phys_error_rate'], n=entry_size)

        output = decoder.decode(entry)

        # Check if output is the all 0 codeword
        if output is None:
            failures.append(1)
        elif np.allclose(output, np.zeros(len(output))):
            failures.append(0)
        else:
            failures.append(1)

    return failures, decoder.times


def run_study(params_file, results_file, codes_path='selected_codes'):
    '''
    Runs a batch of decoding procedures studies for a set of parameters from a
    list of dictionnaries in a file.
    '''
    params_list = utils.get_params_dict_list(params_file)

    # progress bar
    for iter, param_set in enumerate(params_list):
        print('[Plotting point '+str(iter+1)+' of '+str(len(params_list))+']')
        # run decoding procedure
        failures, run_times = new_run_func(**param_set)
        
        results = {"results": {
            "failure_rt": np.mean(failures),
            "fail_std": np.std(failures),
            "avg_time": np.mean(run_times),
            "time_std": np.std(run_times)
        }}

        data_point = {**param_set, **results}
        utils.save_results(data_point, filename=results_file)


if __name__ == "__main__":
    run_study("batch_params.json", results_file='results.txt')
