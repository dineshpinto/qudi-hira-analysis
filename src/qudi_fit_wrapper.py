# -*- coding: utf-8 -*-
"""
The following code is part of qudiamond-analysis under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright (c) 2022 Dinesh Pinto. See the LICENSE file at the
top-level directory of this distribution and at
<https://github.com/dineshpinto/qudiamond-analysis/>
"""

from typing import Tuple

import numpy as np
import pandas as pd
from lmfit.model import ModelResult

import src.fit_logic as fitlogic

f = fitlogic.FitLogic()


def perform_fit(
        x: pd.Series,
        y: pd.Series,
        fit_function: str,
        estimator: str = "generic",
        dims: str = "1d") -> Tuple[np.ndarray, np.ndarray, ModelResult]:
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    fit = {dims: {'default': {'fit_function': fit_function, 'estimator': estimator}}}

    user_fit = f.validate_load_fits(fit)

    use_settings = {}
    for key in user_fit[dims]["default"]["parameters"].keys():
        use_settings[key] = False
    user_fit[dims]["default"]["use_settings"] = use_settings

    fc = f.make_fit_container("test", dims)
    fc.set_fit_functions(user_fit[dims])
    fc.set_current_fit("default")
    fit_x, fit_y, result = fc.do_fit(x, y)
    return fit_x, fit_y, result


def get_fits(dim: str = "1d") -> list:
    return f.fit_list[dim].keys()
