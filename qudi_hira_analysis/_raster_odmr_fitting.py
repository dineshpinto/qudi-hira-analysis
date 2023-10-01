"""
This file contains the Qudi FitLogic class, which provides all
fitting methods imported from the files in logic/_fitmethods.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

import logging
from collections import OrderedDict

import numpy as np
from lmfit import Model
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import convolve1d

logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)


def constant_function(x, offset):
    return offset


const_model = None


def make_constant_model(prefix=None):
    global const_model
    if not isinstance(prefix, str) and prefix is not None:
        log.warning('The passed prefix <{0}> of type {1} is not a string and cannot be used as '
                    'a prefix and will be ignored for now. Correct that!'.format(prefix,
                                                                                 type(prefix)))
        const_model = Model(constant_function, independent_vars=['x'])
    else:
        const_model = Model(constant_function, independent_vars=['x'], prefix=prefix)

    params = const_model.make_params()

    return const_model, params


def amplitude_function(x, amplitude):
    return amplitude


amp_model = None


def make_amplitude_model(prefix=None):
    global amp_model
    if not isinstance(prefix, str) and prefix is not None:
        print('The passed prefix <{0}> of type {1} is not a string and cannot'
              'be used as a prefix and will be ignored for now. Correct that!'.format(prefix, type(prefix)))
        amp_model = Model(amplitude_function, independent_vars=['x'])
    else:
        amp_model = Model(amplitude_function, independent_vars=['x'], prefix=prefix)

    params = amp_model.make_params()

    return amp_model, params


def physical_lorentzian(x, center, sigma):
    return np.power(sigma, 2) / (np.power((center - x), 2) + np.power(sigma, 2))


lorentz_model = None


def make_lorentzianwithoutoffset_model(prefix=None):
    amplitude_model, params = make_amplitude_model(prefix=prefix)
    global lorentz_model
    if not isinstance(prefix, str) and prefix is not None:
        log.error(
            'The passed prefix <{0}> of type {1} is not a string and'
            'cannot be used as a prefix and will be ignored for now.'
            'Correct that!'.format(prefix, type(prefix)))
        lorentz_model = Model(physical_lorentzian, independent_vars=['x'])
    else:
        lorentz_model = Model(
            physical_lorentzian,
            independent_vars=['x'],
            prefix=prefix)

    full_lorentz_model = amplitude_model * lorentz_model
    params = full_lorentz_model.make_params()

    if prefix is None:
        prefix = ''
    full_lorentz_model.set_param_hint(
        '{0!s}fwhm'.format(prefix),
        expr="2*{0!s}sigma".format(prefix))
    return full_lorentz_model, params


####################################
# Lorentzian model with offset     #
####################################
lorentz_offset_model = None


def make_lorentzian_model(prefix=None):
    lorentzian_model, params = make_lorentzianwithoutoffset_model(prefix=prefix)
    constant_model, params = make_constant_model(prefix=prefix)

    global lorentz_offset_model
    lorentz_offset_model = lorentzian_model + constant_model

    if prefix is None:
        prefix = ''

    lorentz_offset_model.set_param_hint('{0}contrast'.format(prefix),
                                        expr='({0}amplitude/offset)*100'.format(prefix))

    params = lorentz_offset_model.make_params()

    return lorentz_offset_model, params


#################################################
#    Multiple Lorentzian model with offset     #
#################################################

def make_multiplelorentzian_model(no_of_functions=1):
    if no_of_functions == 1:
        multi_lorentz_model, params = make_lorentzian_model()
    else:
        prefix = 'l0_'
        multi_lorentz_model, params = make_lorentzianwithoutoffset_model(prefix=prefix)

        constant_model, params = make_constant_model()
        multi_lorentz_model = multi_lorentz_model + constant_model

        multi_lorentz_model.set_param_hint(
            '{0}contrast'.format(prefix),
            expr='({0}amplitude/offset)*100'.format(prefix))

        for ii in range(1, no_of_functions):
            prefix = 'l{0:d}_'.format(ii)
            multi_lorentz_model += make_lorentzianwithoutoffset_model(prefix=prefix)[0]
            multi_lorentz_model.set_param_hint(
                '{0}contrast'.format(prefix),
                expr='({0}amplitude/offset)*100'.format(prefix))

    params = multi_lorentz_model.make_params()

    return multi_lorentz_model, params


#################################################
#    Double Lorentzian model with offset        #
#################################################

def make_lorentziandouble_model():
    return make_multiplelorentzian_model(no_of_functions=2)


def find_offset_parameter(x_values=None, data=None):
    # lorentzian filter
    mod, params = make_lorentzian_model()

    if len(x_values) < 20.:
        len_x = 5
    elif len(x_values) >= 100.:
        len_x = 10
    else:
        len_x = int(len(x_values) / 10.) + 1

    lorentz = mod.eval(x=np.linspace(0, len_x, len_x), amplitude=1, offset=0.,
                       sigma=len_x / 4., center=len_x / 2.)
    data_smooth = convolve1d(data, lorentz / lorentz.sum(),
                             mode='constant', cval=data.max())

    # finding most frequent value which is supposed to be the offset
    hist = np.histogram(data_smooth, bins=10)
    offset = (hist[1][hist[0].argmax()] + hist[1][hist[0].argmax() + 1]) / 2.

    return data_smooth, offset


def estimate_lorentzian_dip(x_axis, data, params):
    # check if parameters make sense
    # error = self._check_1D_input(x_axis=x_axis, data=data, params=params)
    error = None

    # check if input x-axis is ordered and increasing
    sorted_indices = np.argsort(x_axis)
    if not np.all(sorted_indices == np.arange(len(x_axis))):
        x_axis = x_axis[sorted_indices]
        data = data[sorted_indices]

    data_smooth, offset = find_offset_parameter(x_axis, data)

    # data_level = data-offset
    data_level = data_smooth - offset

    # calculate from the leveled data the amplitude:
    amplitude = data_level.min()

    smoothing_spline = 1  # must be 1<= smoothing_spline <= 5
    fit_function = InterpolatedUnivariateSpline(x_axis, data_level,
                                                k=smoothing_spline)
    numerical_integral = fit_function.integral(x_axis[0], x_axis[-1])

    x_zero = x_axis[np.argmin(data_smooth)]

    # according to the derived formula, calculate sigma. The crucial part is
    # here that the offset was estimated correctly, then the area under the
    # curve is calculated correctly:
    sigma = np.abs(numerical_integral / (np.pi * amplitude))

    # auxiliary variables
    stepsize = x_axis[1] - x_axis[0]
    n_steps = len(x_axis)

    params['amplitude'].set(value=amplitude, max=-1e-12)
    params['sigma'].set(value=sigma, min=stepsize / 2,
                        max=(x_axis[-1] - x_axis[0]) * 10)
    params['center'].set(value=x_zero, min=(x_axis[0]) - n_steps * stepsize,
                         max=(x_axis[-1]) + n_steps * stepsize)
    params['offset'].set(value=offset)

    return error, params


def make_lorentzian_fit(x_axis, data, model, params, units=None, **kwargs):
    try:
        result = model.fit(data, x=x_axis, params=params, **kwargs)
    except:
        result = model.fit(data, x=x_axis, params=params, **kwargs)
        log.error('The 1D lorentzian fit did not work. Error '
                  'message: {0}\n'.format(result.message))

    # Write the parameters to allow human-readable output to be generated
    result_str_dict = OrderedDict()

    if units is None:
        units = ["arb. units"]

    result_str_dict['Position'] = {'value': result.params['center'].value,
                                   'error': result.params['center'].stderr,
                                   'unit': units[0]}

    result_str_dict['Contrast'] = {'value': abs(result.params['contrast'].value),
                                   'error': result.params['contrast'].stderr,
                                   'unit': '%'}

    result_str_dict['FWHM'] = {'value': result.params['fwhm'].value,
                               'error': result.params['fwhm'].stderr,
                               'unit': units[0]}

    result_str_dict['chi_sqr'] = {'value': result.chisqr, 'unit': ''}

    result.result_str_dict = result_str_dict
    return result


def _search_end_of_dip(direction, data, peak_arg, start_arg, end_arg, sigma_threshold, minimal_threshold,
                       make_prints):
    """
    data has to be offset leveled such that offset is substracted
    """
    absolute_min = data[peak_arg]

    if direction == 'left':
        mult = -1
        sigma_arg = start_arg
    elif direction == 'right':
        mult = +1
        sigma_arg = end_arg
    else:
        log.error('No valid direction in search end of peak')
        raise ValueError('No valid direction in search end of peak')
    ii = 0

    # if the minimum is at the end set this as boarder
    if (peak_arg != start_arg and direction == 'left' or
            peak_arg != end_arg and direction == 'right'):
        while True:
            # if no minimum can be found decrease threshold
            if ((peak_arg - ii < start_arg and direction == 'left') or
                    (peak_arg + ii > end_arg and direction == 'right')):
                sigma_threshold *= 0.9
                ii = 0
                if make_prints:
                    log.info('h1 sigma_threshold', sigma_threshold)

            # if the dip is always over threshold the end is as
            # set before
            if abs(sigma_threshold / absolute_min) < abs(minimal_threshold):
                if make_prints:
                    log.info('h2')
                break

            # check if value was changed and search is finished
            if ((sigma_arg == start_arg and direction == 'left') or
                    (sigma_arg == end_arg and direction == 'right')):
                # check if if value is lower as threshold this is the
                # searched value
                if make_prints:
                    log.info('h3')
                if abs(data[peak_arg + (mult * ii)]) < abs(sigma_threshold):
                    # value lower than threshold found - left end found
                    sigma_arg = peak_arg + (mult * ii)
                    if make_prints:
                        log.info('h4')
                    break
            ii += 1

    # in this case the value is the last index and should be search set
    # as right argument
    else:
        if make_prints:
            log.info('neu h10')
        sigma_arg = peak_arg

    return sigma_threshold, sigma_arg


def _search_double_dip(x_axis, data, threshold_fraction=0.3,
                       minimal_threshold=0.01, sigma_threshold_fraction=0.3,
                       make_prints=False):
    if sigma_threshold_fraction is None:
        sigma_threshold_fraction = threshold_fraction

    error = 0

    # first search for absolute minimum
    absolute_min = data.min()
    absolute_argmin = data.argmin()

    # adjust thresholds
    threshold = threshold_fraction * absolute_min
    sigma_threshold = sigma_threshold_fraction * absolute_min

    dip0_arg = absolute_argmin

    # ====== search for the left end of the dip ======

    sigma_threshold, sigma0_argleft = _search_end_of_dip(
        direction='left',
        data=data,
        peak_arg=absolute_argmin,
        start_arg=0,
        end_arg=len(data) - 1,
        sigma_threshold=sigma_threshold,
        minimal_threshold=minimal_threshold,
        make_prints=make_prints)

    if make_prints:
        log.info('Left sigma of main peak: ', x_axis[sigma0_argleft])

    # ====== search for the right end of the dip ======
    # reset sigma_threshold

    sigma_threshold, sigma0_argright = _search_end_of_dip(
        direction='right',
        data=data,
        peak_arg=absolute_argmin,
        start_arg=0,
        end_arg=len(data) - 1,
        sigma_threshold=sigma_threshold_fraction * absolute_min,
        minimal_threshold=minimal_threshold,
        make_prints=make_prints)

    if make_prints:
        log.info('Right sigma of main peak: ', x_axis[sigma0_argright])

    # ======== search for second lorentzian dip ========
    left_index = int(0)
    right_index = len(x_axis) - 1

    mid_index_left = sigma0_argleft
    mid_index_right = sigma0_argright

    # if main first dip covers the whole left side search on the right
    # side only
    if mid_index_left == left_index:
        if make_prints:
            log.info('h11', left_index, mid_index_left, mid_index_right, right_index)
        # if one dip is within the second they have to be set to one
        if mid_index_right == right_index:
            dip1_arg = dip0_arg
        else:
            dip1_arg = data[mid_index_right:right_index].argmin() + mid_index_right

    # if main first dip covers the whole right side search on the left
    # side only
    elif mid_index_right == right_index:
        if make_prints:
            log.info('h12')
        # if one dip is within the second they have to be set to one
        if mid_index_left == left_index:
            dip1_arg = dip0_arg
        else:
            dip1_arg = data[left_index:mid_index_left].argmin()

    # search for peak left and right of the dip
    else:
        while True:
            # set search area excluding the first dip
            left_min = data[left_index:mid_index_left].min()
            left_argmin = data[left_index:mid_index_left].argmin()
            right_min = data[mid_index_right:right_index].min()
            right_argmin = data[mid_index_right:right_index].argmin()

            if abs(left_min) > abs(threshold) and \
                    abs(left_min) > abs(right_min):
                if make_prints:
                    log.info('h13')
                # there is a minimum on the left side which is higher
                # than the minimum on the right side
                dip1_arg = left_argmin + left_index
                break
            elif abs(right_min) > abs(threshold):
                # there is a minimum on the right side which is higher
                # than on left side
                dip1_arg = right_argmin + mid_index_right
                if make_prints:
                    log.info('h14')
                break
            else:
                # no minimum at all over threshold so lowering threshold
                #  and resetting search area
                threshold *= 0.9
                left_index = int(0)
                right_index = len(x_axis) - 1
                mid_index_left = sigma0_argleft
                mid_index_right = sigma0_argright
                if make_prints:
                    log.info('h15')
                # if no second dip can be found set both to same value
                if abs(threshold / absolute_min) < abs(minimal_threshold):
                    if make_prints:
                        log.info('h16')
                    log.warning('Threshold to minimum ratio was too '
                                'small to estimate two minima. So both '
                                'are set to the same value')
                    error = -1
                    dip1_arg = dip0_arg
                    break

    # if the dip is exactly at one of the boarders that means
    # the dips are most probably overlapping
    if dip1_arg in (sigma0_argleft, sigma0_argright):
        # print('Dips are overlapping')
        distance_left = abs(dip0_arg - sigma0_argleft)
        distance_right = abs(dip0_arg - sigma0_argright)
        sigma1_argleft = sigma0_argleft
        sigma1_argright = sigma0_argright
        if distance_left > distance_right:
            dip1_arg = dip0_arg - abs(distance_left - distance_right)
        elif distance_left < distance_right:
            dip1_arg = dip0_arg + abs(distance_left - distance_right)
        else:
            dip1_arg = dip0_arg
        # print(distance_left,distance_right,dip1_arg)
    else:
        # if the peaks are not overlapping search for left and right
        # boarder of the dip

        # ====== search for the right end of the dip ======
        sigma_threshold, sigma1_argleft = _search_end_of_dip(
            direction='left',
            data=data,
            peak_arg=dip1_arg,
            start_arg=0,
            end_arg=len(data) - 1,
            sigma_threshold=sigma_threshold_fraction * absolute_min,
            minimal_threshold=minimal_threshold,
            make_prints=make_prints)

        # ====== search for the right end of the dip ======
        sigma_threshold, sigma1_argright = _search_end_of_dip(
            direction='right',
            data=data,
            peak_arg=dip1_arg,
            start_arg=0,
            end_arg=len(data) - 1,
            sigma_threshold=sigma_threshold_fraction * absolute_min,
            minimal_threshold=minimal_threshold,
            make_prints=make_prints)

    return error, sigma0_argleft, dip0_arg, sigma0_argright, sigma1_argleft, dip1_arg, sigma1_argright


def estimate_lorentziandouble_dip(x_axis, data, params,
                                  threshold_fraction=0.3,
                                  minimal_threshold=0.01,
                                  sigma_threshold_fraction=0.3):
    # smooth with gaussian filter and find offset:
    data_smooth, offset = find_offset_parameter(x_axis, data)

    # level data:
    data_level = data_smooth - offset

    # search for double lorentzian dip:
    ret_val = _search_double_dip(x_axis, data_level, threshold_fraction,
                                 minimal_threshold,
                                 sigma_threshold_fraction)

    error = ret_val[0]
    sigma0_argleft, dip0_arg, sigma0_argright = ret_val[1:4]
    sigma1_argleft, dip1_arg, sigma1_argright = ret_val[4:7]

    if dip0_arg == dip1_arg:
        lorentz0_amplitude = data_level[dip0_arg] / 2.
        lorentz1_amplitude = lorentz0_amplitude
    else:
        lorentz0_amplitude = data_level[dip0_arg]
        lorentz1_amplitude = data_level[dip1_arg]

    lorentz0_center = x_axis[dip0_arg]
    lorentz1_center = x_axis[dip1_arg]

    smoothing_spline = 1  # must be 1<= smoothing_spline <= 5
    fit_function = InterpolatedUnivariateSpline(x_axis, data_level,
                                                k=smoothing_spline)
    numerical_integral_0 = fit_function.integral(x_axis[sigma0_argleft],
                                                 x_axis[sigma0_argright])

    lorentz0_sigma = abs(numerical_integral_0 / (np.pi * lorentz0_amplitude))

    numerical_integral_1 = numerical_integral_0

    lorentz1_sigma = abs(numerical_integral_1 / (np.pi * lorentz1_amplitude))

    # esstimate amplitude

    stepsize = x_axis[1] - x_axis[0]
    full_width = x_axis[-1] - x_axis[0]
    n_steps = len(x_axis)

    if lorentz0_center < lorentz1_center:
        params['l0_amplitude'].set(value=lorentz0_amplitude, max=-0.01)
        params['l0_sigma'].set(value=lorentz0_sigma, min=stepsize / 2,
                               max=full_width * 4)
        params['l0_center'].set(value=lorentz0_center,
                                min=(x_axis[0]) - n_steps * stepsize,
                                max=(x_axis[-1]) + n_steps * stepsize)
        params['l1_amplitude'].set(value=lorentz1_amplitude, max=-0.01)
        params['l1_sigma'].set(value=lorentz1_sigma, min=stepsize / 2,
                               max=full_width * 4)
        params['l1_center'].set(value=lorentz1_center,
                                min=(x_axis[0]) - n_steps * stepsize,
                                max=(x_axis[-1]) + n_steps * stepsize)
    else:
        params['l0_amplitude'].set(value=lorentz1_amplitude, max=-0.01)
        params['l0_sigma'].set(value=lorentz1_sigma, min=stepsize / 2,
                               max=full_width * 4)
        params['l0_center'].set(value=lorentz1_center,
                                min=(x_axis[0]) - n_steps * stepsize,
                                max=(x_axis[-1]) + n_steps * stepsize)
        params['l1_amplitude'].set(value=lorentz0_amplitude, max=-0.01)
        params['l1_sigma'].set(value=lorentz0_sigma, min=stepsize / 2,
                               max=full_width * 4)
        params['l1_center'].set(value=lorentz0_center,
                                min=(x_axis[0]) - n_steps * stepsize,
                                max=(x_axis[-1]) + n_steps * stepsize)

    params['offset'].set(value=offset)

    return error, params


def make_lorentziandouble_fit(x_axis, data, model, params, units=None, **kwargs):
    try:
        result = model.fit(data, x=x_axis, params=params, **kwargs)
    except:
        result = model.fit(data, x=x_axis, params=params, **kwargs)
        log.error('The double lorentzian fit did not '
                  f'work: {result.message}')

    # Write the parameters to allow human-readable output to be generated
    result_str_dict = OrderedDict()

    if units is None:
        units = ["arb. u."]

    result_str_dict['Position 0'] = {'value': result.params['l0_center'].value,
                                     'error': result.params['l0_center'].stderr,
                                     'unit': units[0]}

    result_str_dict['Position 1'] = {'value': result.params['l1_center'].value,
                                     'error': result.params['l1_center'].stderr,
                                     'unit': units[0]}

    try:
        result_str_dict['Splitting'] = {'value': (result.params['l1_center'].value -
                                                  result.params['l0_center'].value),
                                        'error': (result.params['l0_center'].stderr +
                                                  result.params['l1_center'].stderr),
                                        'unit': units[0]}
    except TypeError:
        result_str_dict['Splitting'] = {'value': (result.params['l1_center'].value -
                                                  result.params['l0_center'].value),
                                        'error': np.inf,
                                        'unit': units[0]}

    result_str_dict['Contrast 0'] = {'value': abs(result.params['l0_contrast'].value),
                                     'error': result.params['l0_contrast'].stderr,
                                     'unit': '%'}

    result_str_dict['Contrast 1'] = {'value': abs(result.params['l1_contrast'].value),
                                     'error': result.params['l1_contrast'].stderr,
                                     'unit': '%'}

    result_str_dict['FWHM 0'] = {'value': result.params['l0_fwhm'].value,
                                 'error': result.params['l0_fwhm'].stderr,
                                 'unit': units[0]}

    result_str_dict['FWHM 1'] = {'value': result.params['l1_fwhm'].value,
                                 'error': result.params['l1_fwhm'].stderr,
                                 'unit': units[0]}

    result_str_dict['chi_sqr'] = {'value': result.chisqr, 'unit': ''}

    result.result_str_dict = result_str_dict
    return result
