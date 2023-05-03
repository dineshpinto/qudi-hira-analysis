import unittest
from pathlib import Path

from qudi_hira_analysis import DataHandler

dh = DataHandler(
    data_folder=Path(__file__).parent.resolve() / "data",
    figure_folder=Path(__file__).parent.resolve() / "figures",
)


class TestFitting(unittest.TestCase):
    def test_autocorrelation_antibunching_fit(self):
        autocorrs = dh.load_measurements(measurement_str="Autocorrelation")
        autocorr = autocorrs['20230306-1732-05']

        autocorr.data["Time (ns)"] = autocorr.data["Time (ps)"] * 1e-3
        fit_x, fit_y, _ = dh.fit(x="Time (ns)", y="g2(t) norm", data=autocorr.data,
                                 fit_function=dh.fit_function.antibunching)

        self.assertAlmostEqual(fit_x[0], -1000)
        self.assertAlmostEqual(fit_y[0], 1.00109277)

    def test_pulsedodmr_lorentziandouble_fit(self):
        odmr_list = dh.load_measurements(measurement_str="ODMR", pulsed=True)
        odmr = odmr_list["20220315-2050-39"]

        x_fit, y_fit, _ = dh.fit(x="Controlled variable(Hz)", y="Signal", data=odmr.data,
                                 fit_function=dh.fit_function.lorentziandouble)

        self.assertAlmostEqual(x_fit[0], 2850000000.0)
        self.assertAlmostEqual(y_fit[0], 1.1015461033795304)

    def test_pusledrabi_sineexponentialdecay_fit(self):
        rabi_list = dh.load_measurements(measurement_str="Rabi", pulsed=True)
        rabi = rabi_list['20220316-1434-53']
        x_fit, y_fit, _ = dh.fit(x="Controlled variable(s)", y="Signal", data=rabi.data,
                                 fit_function=dh.fit_function.sineexponentialdecay)

        self.assertAlmostEqual(x_fit[0], 0)
        self.assertAlmostEqual(y_fit[0], 1.1042166220436456)

    def test_saturation_hyperbolicsaturation_fit(self):
        sat = dh.read_excel(dh.data_folder_path / "saturation_curve.xlsx", skiprows=1)
        x_fit, y_fit, _ = dh.fit(x="Power", y="Counts", data=sat,
                                 fit_function=dh.fit_function.hyperbolicsaturation)
        self.assertAlmostEqual(x_fit[0], 0.83)
        self.assertAlmostEqual(y_fit[0], 54533.66570552715)


if __name__ == '__main__':
    unittest.main()
