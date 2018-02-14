#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation

Example:
        python local.py '{"input":
                                {"covariates":
                                    [[2,3],[3,4],[7,8],[7,5],[9,8]],
                                "dependents":
                                    [6,7,8,5,6],
                                "lambda":0
                                },
                            "cache":{}
                            }'
"""
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import sys
import regression as reg
import statsmodels.api as sm


def local_1(args, computation_phase):
    """Computes local beta vector and local fit statistics

    Args:
        args (dictionary) : {"input":
                                {"covariates": ,
                                 "dependents": ,
                                 "lambda":
                                },
                            "cache": {}
                            }

        computation_phase (string) : Field specifying which part (local/
                                     remote) of the decentralized computation
                                     has been performed last
                                     In this case, it has to be empty
    Returns:
        computation_output(json) : {
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

    Comments:
        Step 1 : Generate the local beta_vector
        Step 2 : Generate the local fit statistics
                    r^2 : goodness of fit/coefficient of determination
                          Given as 1 - (SSE/SST)
                          where  SSE = Sum Squared of Errors
                                 SST = Total Sum of Squares
                    t   : t-statistic is the coefficient divided by
                          its standard error.
                          Given as beta/std.err(beta)
                    p   : two-tailed p-value (The p-value is the probability of
                          seeing a result as extreme as the one you are
                          getting (a t value as large as yours)
                          in a collection of random data in which
                          the variable had no effect.)
        Step 3 : Compute mean_y_local and length of target values
    """
    input_list = args['input']
    X = input_list['covariates']
    y = input_list['dependents']
    lamb = input_list['lambda']
#    biased_X = np.insert(X, 0, 1, axis=1)
    biased_X = sm.add_constant(X)

    beta_vector = reg.one_shot_regression(biased_X, y, lamb)

    r_squared = reg.r_square(biased_X, y, beta_vector)
    ts_beta = reg.t_value(biased_X, y, beta_vector)
    dof = len(y) - len(beta_vector)
    ps_beta = reg.t_to_p(ts_beta, dof)

    computation_output_dict = {
        'output': {
            'beta_vector_local': beta_vector.tolist(),
            'r_2_local': r_squared,
            'ts_local': ts_beta.tolist(),
            'ps_local': ps_beta,
            'mean_y_local': np.mean(y),
            'count_local': len(y),
            'computation_phase': computation_phase
        },
        'cache': {
            'covariates': X,
            'dependents': y,
            'lambda': lamb
        },
        'success': True
    }

#    return json.dumps(
#        computation_output_dict,
#        sort_keys=True,
#        indent=4,
#        separators=(',', ': '))

    return json.dumps(computation_output_dict)


def local_2(args, computation_phase):
    """Calculates the SSE_local, SST_local and varX_matrix_local

    Args:
        args (dictionary): {
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
        computation_phase (string): A field specifying which part
                                    (local/remote) of the decentralized
                                    computation has been performed last
                                    In this case, it has to be remote_1

    Returns:
        computation_output (json): {
                                        'output': {
                                            'SSE_local': ,
                                            'SST_local': ,
                                            'varX_matrix_local': ,
                                            'computation_phase':
                                        }
                                    }

    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local
    """
    cache_list = args['cache']
    input_list = args['input']

    X = cache_list['covariates']
    y = cache_list['dependents']
    biased_X = np.insert(X, 0, 1, axis=1)

    avg_beta_vector = input_list['avg_beta_vector']
    mean_y_global = input_list['mean_y_global']

    SSE_local = reg.sum_squared_error(biased_X, y, avg_beta_vector)
    SST_local = np.sum(np.square(y - mean_y_global))
    varX_matrix_local = np.dot(biased_X.T, biased_X)

    computation_output_dict = {
        'output': {
            'SSE_local': SSE_local,
            'SST_local': SST_local,
            'varX_matrix_local': varX_matrix_local,
            'computation_phase': computation_phase
        }
    }

#    return json.dumps(
#        computation_output_dict,
#        sort_keys=True,
#        indent=4,
#        separators=(',', ': '))
    return json.dumps(
        computation_output_dict)


if __name__ == '__main__':
    parsed_args = json.loads(sys.argv[1])

    if 'computation_phase' not in parsed_args.keys():
        computation_phase = 'local_1'
        computation_output = local_1(parsed_args, computation_phase)
        sys.stdout.write(computation_output)
    elif parsed_args['computation_phase'] == 'remote_1':
        computation_phase = 'local_2'
        computation_output = local_2(parsed_args, computation_phase)
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Invalid value for computation_phase')
