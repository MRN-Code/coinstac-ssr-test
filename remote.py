#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import argparse
import json
import sys
import scipy as sp
import numpy as np
import regression as reg


def remote_1(args, computation_phase):
    """
    Args:
        args (dictionary): {
                                'output': {
                                    'beta_vector_local': ,
                                    'r_2_local': ,
                                    'ts_local': ,
                                    'ps_local': ,
                                    'mean_y_local': ,
                                    'count_local': ,
                                    'computation_phase':
                                },
                                'cache': {
                                    'covariates': ,
                                    'dependents': ,
                                    'lambda':
                                }
                            }

        computation_phase (string): field specifying which part (local/remote)
                                   of the decentralized computation has been
                                   performed last
                                   In this case, it has to be empty

    Returns:
        computation_output (json) : {
                                    'cache': {
                                        'avg_beta_vector': ,
                                        'mean_y_global': ,
                                        'dof_global': ,
                                        'dof_local': ,
                                        'beta_vector_local': ,
                                        'r_2_local': ,
                                        'ts_local': ,
                                        'ps_local':
                                    },
                                    'output': {
                                        'avg_beta_vector': ,
                                        'mean_y_global': ,
                                        'computation_phase':
                                    }
                                }

    Comments:
        Step 1: Calculate the averaged beta vector, mean_y_global & dof_global
        Step 2: Retrieve the local fit statistics and save them in the cache
    """
    input_list = args['input']

    # Step 1
    avg_beta_vector = np.mean(
        [site['beta_vector_local'] for site in input_list])

    mean_y_local = [site['mean_y_local'] for site in input_list]
    count_y_local = [site['count_local'] for site in input_list]
    mean_y_global = np.average(mean_y_local, weights=count_y_local)

    dof_global = np.sum(count_y_local) - len(avg_beta_vector)

    # Step 2
    beta_vector_local = [site['beta_vector_local'] for site in input_list]
    dof_local = [
        site['count_local'] - len(avg_beta_vector) for site in input_list
    ]
    r_2_local = [site['r_2_local'] for site in input_list]
    ts_local = [site['ts_local'] for site in input_list]
    ps_local = [site['ps_local'] for site in input_list]

    computation_output_dict = {
        'cache': {
            'avg_beta_vector': avg_beta_vector.tolist(),
            'mean_y_global': mean_y_global,
            'dof_global': dof_global,
            'dof_local': dof_local,
            'beta_vector_local': beta_vector_local,
            'r_2_local': r_2_local,
            'ts_local': ts_local,
            'ps_local': ps_local
        },
        'output': {
            'avg_beta_vector': avg_beta_vector.tolist(),
            'mean_y_global': mean_y_global,
            'computation_phase': computation_phase
        }
    }

    return json.dumps(
        computation_output_dict,
        sort_keys=True,
        indent=4,
        separators=(',', ': '))


def remote_2(args, computation_phase):
    """
    # calculate the global model fit statistics, r_2_global, ts_global,
    # ps_global
    Args:
        args (dictionary): {
                                'output': {
                                    'SSE_local': ,
                                    'SST_local': ,
                                    'varX_matrix_local': ,
                                    'computation_phase':
                                }
                            }

        computation_phase (String): field specifying which part (local/remote)
                                   of the decentralized computation has been
                                   performed last
                                   In this case, it has to be empty
    Returns:
        computation_output (json) : {
                                        'output': {
                                            'avg_beta_vector': ,
                                            'beta_vector_local': ,
                                            'r_2_global': ,
                                            'ts_global': ,
                                            'ps_global': ,
                                            'r_2_local': ,
                                            'ts_local': ,
                                            'ps_local': ,
                                            'dof_global': ,
                                            'dof_local': ,
                                            'complete':
                                        }
                                    }
    """
    cache_list = args['cache']
    input_list = args['input']
    avg_beta_vector = cache_list['avg_beta_vector']
    dof_global = cache_list['dof_global']

    SSE_global = np.sum([site['SSE_Local'] for site in input_list])
    SST_global = np.sum([site['SST_Local'] for site in input_list])
    varX_matrix_global = np.sum(
        [site['varX_matrix_Local'] for site in input_list])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / dof_global
    var_covar_beta_global = MSE * sp.linalg.inv(varX_matrix_global)
    se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
    ts_global = avg_beta_vector / se_beta_global
    ps_global = reg.t_to_p(ts_global, dof_global)

    computation_output_dict = {
        'output': {
            'avg_beta_vector': cache_list['avg_beta_vector'],
            'beta_vector_local': cache_list['beta_vector_local'],
            'r_2_global': r_squared_global,
            'ts_global': ts_global,
            'ps_global': ps_global,
            'r_2_local': cache_list['r_2_local'],
            'ts_local': cache_list['ts_local'],
            'ps_local': cache_list['ps_local'],
            'dof_global': cache_list['dof_global'],
            'dof_local': cache_list['dof_local'],
            'complete': True
        }
    }

    return json.dumps(
        computation_output_dict,
        sort_keys=True,
        indent=4,
        separators=(',', ': '))


if __name__ == '__main__':
#    parser = argparse.ArgumentParser(description='help read in coinstac \
#                                     input from local node')
#    parser.add_argument('--run', type=json.loads, help='grab coinstac args')
#    args = parser.parse_args()
#    parsed_args = args.run

    args = json.loads(sys.argv[1])
    parsed_args = args['input']

    # *******block 1*********
    if input_list[0]['computation_phase'] == 'local_1':
        computation_phase = 'remote_1'
        computation_output = remote_1(parsed_args, computation_phase)
        sys.stdout.write(computation_output)

    # *******block 2********#
    elif input_list[0]['computation_phase'] == 'local_2':
        computation_phase = 'remote_2'
        # step 1 calculate the global model fit statistics,
        # r_2_global, t_global, p_global
        computation_output = remote_2(parsed_args, computation_phase)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Errors occurred")
