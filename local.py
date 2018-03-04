#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script includes the local computations for single-shot ridge
regression with decentralized statistic calculation

Example:
    python local.py '{"input":
                        {"covariates": [[2,3],[3,4],[7,8],[7,5],[9,8]],
                         "dependents": [6,7,8,5,6],
                         "lambda": 0
                         },
                     "cache": {}
                     }'
"""
import json
import numpy as np
import os
import sys
import regression as reg
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def local_1(args):
    """Computes local beta vector

    Args:
        args (dictionary) : {"input": {
                                "covariates": ,
                                 "data": ,
                                 "lambda": ,
                                },
                            "cache": {}
                            }

    Returns:
        computation_output(json) : {"output": {
                                        "beta_vector_local": ,
                                        "mean_y_local": ,
                                        "count_local": ,
                                        "computation_phase":
                                        },
                                    "cache": {
                                        "covariates": ,
                                        "dependents": ,
                                        "lambda":
                                        }
                                    }

    Comments:
        Step 1 : Generate the local beta_vector
        Step 2 : Compute mean_y_local and length of target values

    """
    input_list = args["input"]
    X_list = input_list["covariates"]
    y_list = input_list["data"]

    X, y_files = [], []

    for i in range(1, len(X_list[0][0])):
        y_files.append(X_list[0][0][i][0])
        X.append(X_list[0][0][i][1:])

    X = [[int(i[0] == 'True'), float(i[1])] for i in X]

    y = []
    dependents = y_list[1][0]
    file_paths = y_list[0][0]
    for file in file_paths:
        if file.split('-')[-1] in y_files:
            with open(
                    os.path.join(args["state"]["baseDirectory"],
                                 file.split('-')[-1])) as fh:
                for line in fh:
                    if line.startswith(dependents[0]):
                        y.append(float(line.split('\t')[1]))

    lamb = input_list["lambda"]
    biased_X = sm.add_constant(X)
    beta_vector = reg.one_shot_regression(biased_X, y, lamb)

    computation_output = {
        "output": {
            "beta_vector_local": beta_vector.tolist(),
            "mean_y_local": np.mean(y),
            "count_local": len(y),
            "computation_phase": 'local_1'
        },
        "cache": {
            "covariates": X,
            "dependents": y,
            "lambda": lamb
        }
    }

    return json.dumps(computation_output)


def local_2(args):
    """Computes the SSE_local, SST_local and varX_matrix_local

    Args:
        args (dictionary): {"input": {
                                "avg_beta_vector": ,
                                "mean_y_global": ,
                                "computation_phase":
                                },
                            "cache": {
                                "covariates": ,
                                "dependents": ,
                                "lambda": ,
                                "dof_local": ,
                                }
                            }

    Returns:
        computation_output (json): {"output": {
                                        "SSE_local": ,
                                        "SST_local": ,
                                        "varX_matrix_local": ,
                                        "computation_phase":
                                        }
                                    }

    Comments:
        After receiving  the mean_y_global, calculate the SSE_local,
        SST_local and varX_matrix_local

    """
    cache_list = args["cache"]
    input_list = args["input"]

    X = cache_list["covariates"]
    y = cache_list["dependents"]
    biased_X = sm.add_constant(X)

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    SSE_local = reg.sum_squared_error(biased_X, y, avg_beta_vector)
    SST_local = np.sum(np.square(np.subtract(y, mean_y_global)))
    varX_matrix_local = np.dot(biased_X.T, biased_X)

    computation_output = {
        "output": {
            "SSE_local": SSE_local,
            "SST_local": SST_local,
            "varX_matrix_local": varX_matrix_local.tolist(),
            "computation_phase": 'local_2'
        },
        "cache": {}
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.argv[1])
    phase_key = list(reg.listRecursive(parsed_args, 'computation_phase'))

    if not phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "remote_1" in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
