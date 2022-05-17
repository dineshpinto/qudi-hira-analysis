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


def decibelm_to_watts(dbm_value: float) -> float:
    """ 
    Convert dBm to Watts. 
    
     1 W = 10^((1 dBm - 30) / 10)

    Args:
        dbm_value: value in dBm
    
    """
    
    return 10 ** ((dbm_value - 30) / 10)


def format_exponent_as_str(num: float, decimals: int = 2) -> str:
    """
    Format an exponent as a LaTeX string
    e.g. 0.0001 will be formatted as $1.0 \times 10^{-4}$

    Args:
        num: number to format
        decimals: number of decimals to keep

    Returns:
        formatted_str: LaTeX formatted string
    """
    count = 0

    if num > 1:
        while num >= 10:
            num /= 10
            count += 1
    else:
        while num < 1:
            num *= 10
            count -= 1

    formatted_str = r"${{ {} }} \times 10^{{ {} }}$".format(round(num, decimals), count)
    return formatted_str
