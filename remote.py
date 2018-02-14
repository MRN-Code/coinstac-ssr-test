#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the remote computations for single-shot ridge
regression with decentralized statistic calculation
"""
import json
import sys
import scipy as sp
import numpy as np
import regression as reg


def remote_1(args):
    """
    Args:
        args (dictionary): {'input': {
                                'beta_vector_local': ,
                                'mean_y_local': ,
                                'count_local': ,
                                'computation_phase':
                                },
                            'cache': {}
                            }

    Returns:
        computation_output (json) : {'output': {
                                        'avg_beta_vector': ,
                                        'mean_y_global': ,
                                        'computation_phase':
                                        },
                                    'cache': {
                                        'avg_beta_vector': ,
                                        'mean_y_global': ,
                                        'dof_global': ,
                                        },
                                    }

    Comments:
        Step 1: Calculate the averaged beta vector, mean_y_global & dof_global
        Step 2: Retrieve the local fit statistics and save them in the cache
    """

    input_list = args["input"]

    avg_beta_vector = np.mean(
        [input_list[site]["beta_vector_local"] for site in input_list], axis=0)

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [input_list[site]["count_local"] for site in input_list]
    mean_y_global = np.average(mean_y_local, weights=count_y_local)

    dof_global = sum(count_y_local) - len(avg_beta_vector)

    computation_output_dict = {
        "output": {
            "avg_beta_vector": avg_beta_vector.tolist(),
            "mean_y_global": mean_y_global,
            "computation_phase": 'remote_1'
        },
        "cache": {
            "avg_beta_vector": avg_beta_vector.tolist(),
            "mean_y_global": mean_y_global,
            "dof_global": dof_global
        },
    }

    return json.dumps(computation_output_dict)


def remote_2(args):
    """
    # calculate the global model fit statistics, r_2_global, ts_global,
    # ps_global
    Args:
        args (dictionary): {'output': {
                                'SSE_local': ,
                                'SST_local': ,
                                'varX_matrix_local': ,
                                'computation_phase':
                                },
                            'cache':{},
                            'success':
                            }

    Returns:
        computation_output (json) : {'output': {
                                        'avg_beta_vector': ,
                                        'beta_vector_local': ,
                                        'r_2_global': ,
                                        'ts_global': ,
                                        'ps_global': ,
                                        'dof_global':
                                        }
                                    }
    """
    cache_list = args["cache"]
    input_list = args["input"]
    avg_beta_vector = cache_list["avg_beta_vector"]
    dof_global = cache_list["dof_global"]

    SSE_global = np.sum([input_list[site]["SSE_local"] for site in input_list])
    SST_global = np.sum([input_list[site]["SST_local"] for site in input_list])
    varX_matrix_global = sum([
        np.array(input_list[site]["varX_matrix_local"]) for site in input_list
    ])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / dof_global
    var_covar_beta_global = MSE * sp.linalg.inv(varX_matrix_global)
    se_beta_global = np.sqrt(var_covar_beta_global.diagonal())
    ts_global = avg_beta_vector / se_beta_global
    ps_global = reg.t_to_p(ts_global, dof_global)

    computation_output_dict = {
        "output": {
            "avg_beta_vector": cache_list["avg_beta_vector"],
            "r_2_global": r_squared_global,
            "ts_global": ts_global.tolist(),
            "ps_global": ps_global,
            "dof_global": cache_list["dof_global"]
        },
        "computation_phase": 'remote_2',
        "success": True
    }

    return json.dumps(computation_output_dict)


if __name__ == '__main__':

    parsed_args = json.loads(sys.argv[1])

    if parsed_args["input"]["local0"]["computation_phase"] == 'local_1':
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif parsed_args["input"]["local0"]["computation_phase"] == 'local_2':
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error Occurred at Remote")
