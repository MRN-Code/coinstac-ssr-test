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
import sys
import regression as reg
import warnings
from itertools import chain

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statsmodels.api as sm


def local_1(args):
    """Computes local beta vector

    Args:
        args (dictionary) : {"input": {
                                "covariates": ,
                                 "dependents": ,
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
    X = input_list["covariates"]
    y = input_list["dependents"]
    lamb = input_list["lambda"]

    biased_X = map(lambda x: sm.add_constant(x, has_constant='raise'), X)
    all_beta = list(
        map(lambda x: reg.one_shot_regression(x, y, lamb).tolist(), biased_X))

    computation_output = {
        "output": {
            "beta_vector_local": all_beta,
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

    avg_beta_vector = input_list["avg_beta_vector"]
    mean_y_global = input_list["mean_y_global"]

    SSE_local_enigma, SST_local_enigma, varX_matrix_local_enigma = [], [], []
    for index, X_ in enumerate(X):
        biased_X = sm.add_constant(X_)
        SSE_local = reg.sum_squared_error(biased_X, y, avg_beta_vector[index])
        SST_local = np.sum(np.square(np.subtract(y, mean_y_global)))
        varX_matrix_local = np.dot(biased_X.T, biased_X)

        SSE_local_enigma.append(SSE_local)
        SST_local_enigma.append(SST_local)
        varX_matrix_local_enigma.append(varX_matrix_local.tolist())

    computation_output = {
        "output": {
            "SSE_local": SSE_local_enigma,
            "SST_local": SST_local_enigma,
            "varX_matrix_local": varX_matrix_local_enigma,
            "computation_phase": 'local_2'
        },
        "cache": {}
    }

    return json.dumps(computation_output)


def get_all_keys(current_dict):
    children = []
    for k in current_dict:
        yield k
        if isinstance(current_dict[k], dict):
            children.append(get_all_keys(current_dict[k]))
    for k in chain.from_iterable(children):
        yield k


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())

    if "computation_phase" not in list(get_all_keys(parsed_args)):
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif "computation_phase" in list(get_all_keys(parsed_args)):
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
