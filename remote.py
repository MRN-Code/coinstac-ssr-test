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
    """Computes the global beta vector, mean_y_global & dof_global

    Args:
        args (dictionary): {"input": {
                                "beta_vector_local": list/array,
                                "mean_y_local": list/array,
                                "count_local": int,
                                "computation_phase": string
                                },
                            "cache": {}
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": list,
                                        "mean_y_global": ,
                                        "computation_phase":
                                        },
                                    "cache": {
                                        "avg_beta_vector": ,
                                        "mean_y_global": ,
                                        "dof_global":
                                        },
                                    }

    """
    input_list = args["input"]

    avg_beta_vector = np.mean(
        [input_list[site]["beta_vector_local"] for site in input_list], axis=0)

    mean_y_local = [input_list[site]["mean_y_local"] for site in input_list]
    count_y_local = [input_list[site]["count_local"] for site in input_list]
    mean_y_global = np.average(mean_y_local, weights=count_y_local)

    dof_global = sum(count_y_local) - len(avg_beta_vector)

    computation_output = {
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

    return json.dumps(computation_output)


def remote_2(args):
    """
    Computes the global model fit statistics, r_2_global, ts_global, ps_global

    Args:
        args (dictionary): {"input": {
                                "SSE_local": ,
                                "SST_local": ,
                                "varX_matrix_local": ,
                                "computation_phase":
                                },
                            "cache":{},
                            }

    Returns:
        computation_output (json) : {"output": {
                                        "avg_beta_vector": ,
                                        "beta_vector_local": ,
                                        "r_2_global": ,
                                        "ts_global": ,
                                        "ps_global": ,
                                        "dof_global":
                                        },
                                    "success":
                                    }
    Comments:
        Generate the local fit statistics
            r^2 : goodness of fit/coefficient of determination
                    Given as 1 - (SSE/SST)
                    where   SSE = Sum Squared of Errors
                            SST = Total Sum of Squares
            t   : t-statistic is the coefficient divided by its standard error.
                    Given as beta/std.err(beta)
            p   : two-tailed p-value (The p-value is the probability of
                  seeing a result as extreme as the one you are
                  getting (a t value as large as yours)
                  in a collection of random data in which
                  the variable had no effect.)

    """
    input_list = args["input"]

    cache_list = args["cache"]
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

    computation_output = {
        "output": {
            "avg_beta_vector": cache_list["avg_beta_vector"],
            "r_2_global": r_squared_global,
            "ts_global": ts_global.tolist(),
            "ps_global": ps_global,
            "dof_global": cache_list["dof_global"]
        },
        "success": True
    }

    return json.dumps(computation_output)


if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())

    if parsed_args["input"]["local0"]["computation_phase"] == 'local_1':
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif parsed_args["input"]["local0"]["computation_phase"] == 'local_2':
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
