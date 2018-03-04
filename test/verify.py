#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:53:47 2018

@author: Harshvardhan
@notes: Contains code to independently verify the results of the single-shot
        regression where 1 client implies pooled regression
"""

import statsmodels.api as sm

# Example 1 - corresponds to one inputspec.json (3rd-Ventricle)
X = [[0, 22], [1, 47], [0, 56], [1, 73]]
y = [2115.49, 4019.55, 4934.63, 9360.22]
lamb = 0.

X = sm.add_constant(X)


# model = sm.OLS(y, X).fit() is equivalent to the following line
# when lamb=0. and is basic regression
model1 = sm.OLS(y, X).fit_regularized(alpha=lamb, L1_wt=0.)
print(model1.params)
