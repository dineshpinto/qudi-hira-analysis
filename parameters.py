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

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    # Name of computer
    remote_computer_name: str = "PCKK022"
    # The code automatically detects whether kernix is connected remotely or not
    # Use when connected remotely
    kernix_remote_datafolder: str = os.path.join("\\\\kernix", "qudiamond", "Data")
    output_figure_remote_folder: str = ("C:/", "Nextcloud", "Data_Analysis")
    # Use when connected directly
    kernix_local_datafolder: str = os.path.join("Z:/", "Data")
    output_figure_local_folder: str = os.path.join("Z:/", "Data_Analysis")
