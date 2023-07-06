from pathlib import Path
from unittest import TestCase

from qudi_hira_analysis import DataHandler


class TestLoading(TestCase):
    def setUp(self) -> None:
        self.dh = DataHandler(
            data_folder=Path(__file__).parent.resolve() / "data",
            figure_folder=Path(__file__).parent.resolve() / "figures",
        )

    def test_autocorrelation_load(self):
        autocorrs = self.dh.load_measurements(measurement_str="Autocorrelation")
        autocorr = autocorrs['20230306-1732-05']

        self.assertAlmostEqual(autocorr.data["Time (ps)"][0], -1.000000000000000e+06)
        self.assertAlmostEqual(autocorr.data["g2(t) norm"][0], 8.971149729207432e-01)

    def test_pulsedmeasurement_load(self):
        odmr_list = self.dh.load_measurements(measurement_str="ODMR", pulsed=True)
        odmr = odmr_list["20220315-2050-39"]

        self.assertAlmostEqual(odmr.data["Controlled variable(Hz)"][0], 2.850000000000000e+09)
        self.assertAlmostEqual(odmr.data["Signal"][0], 1.091035609573383e+00)

    def test_spectrometer_load(self):
        spectrometry = self.dh.load_measurements(measurement_str="Spectrometry")
        spectrum = spectrometry['20230306-2324-19']

        self.assertAlmostEqual(spectrum.data["wavelength"][0], 3.472799682617187e-07)
        self.assertAlmostEqual(spectrum.data["signal"][0], 1.694000000000000e+03)
        self.assertAlmostEqual(spectrum.data["corrected"][0], -3.000000000000000e+00)

    def test_frq_sweep_load(self):
        frq_sweeps = self.dh.load_measurements(measurement_str="frq-sweep", qudi=False)
        # Generally this would be the exact timestamp
        # But since the timestamp is not stored in the filename,
        # we have to use the first key of the dictionary
        frq_sweep = frq_sweeps[list(frq_sweeps)[0]]
        self.assertAlmostEqual(frq_sweep.data["Frequency Shift (Hz)"][0], -38.1000)
        self.assertAlmostEqual(frq_sweep.data["Amplitude (m)"][0], 5.205450e-10)
        self.assertEqual(frq_sweep.params["f_res (Hz)"], 30281.5211)
        self.assertEqual(frq_sweep.params["Q"], 1161.0)

    def test_nvpl_scan_load(self):
        pixel_scans = self.dh.load_measurements(measurement_str="PixelScanner")
        fwd, bwd = pixel_scans["20230302-1359-15"].data

        self.assertEqual(fwd.channel, "Forward")
        self.assertEqual(bwd.channel, "Backward")
        self.assertAlmostEqual(fwd.pixels[0][0], 74261.87601444275)

    def test_nanonis_afm_load(self):
        afm_scans = self.dh.load_measurements(measurement_str="Scan", extension=".sxm", qudi=False)
        afm = afm_scans[list(afm_scans)[0]].data

        topo = afm.get_channel("Z")
        self.assertAlmostEqual(topo.pixels[0][0], 9.72290422396327e-07)

        exc = afm.get_channel("Excitation")
        self.assertAlmostEqual(exc.pixels[0][0], 1.499999761581421)

    def test_bruker_mfm_load(self):
        bruker_measurements = self.dh.load_measurements(measurement_str="", extension=".001", qudi=False)

        bruker_data = bruker_measurements["20230324-1256-11"].data
        mfm = bruker_data.get_channel("Phase", mfm=True)

        self.assertEqual(mfm.channel, "Phase")
        self.assertEqual(mfm.type, "Bruker MFM")

    def test_pys_load(self):
        pys_measurements = self.dh.load_measurements(measurement_str="ndmin", extension=".pys", qudi=False)
        pys = pys_measurements[list(pys_measurements)[0]].data

        self.assertAlmostEqual(pys["time_bins"][0], -298.0)
        self.assertAlmostEqual(pys["counts"][0], 1210)

    def test_confocal_load(self):
        confocal_scans = self.dh.load_measurements(measurement_str="Confocal")
        confocal = confocal_scans["20230330-1113-04"].data
        self.assertAlmostEqual(confocal.iloc[0][0], 600.0)

    def test_temperature_monitoring_load(self):
        temps = self.dh.load_measurements(measurement_str="temperature-monitoring", qudi=False)
        temp = temps[list(temps)[0]].data
        self.assertIn("Tip Holder", temp.columns)
        self.assertAlmostEqual(temp.iloc[0][0], 119.403)

    def test_measurement_dataclass(self):
        odmr_list = self.dh.load_measurements(measurement_str="ODMR", pulsed=True)
        odmr = odmr_list["20220315-2050-39"]

        self.assertAlmostEqual(odmr.data["Controlled variable(Hz)"][0], 2.850000000000000e+09)
        self.assertAlmostEqual(odmr.data["Signal"][0], 1.091035609573383e+00)

        sig, err = self.dh.analyse_mean(odmr.pulsed.laser_pulses.data)
        self.assertAlmostEqual(sig[0], 694.41)
        self.assertAlmostEqual(err[0], 1.8633437686052456)

        sig, err = self.dh.analyse_mean_norm(odmr.pulsed.laser_pulses.data)
        self.assertAlmostEqual(sig[0], 1.1218563353113091)
        self.assertAlmostEqual(err[0], 0.0033309708217416473)

        sig, err = self.dh.analyse_mean_reference(odmr.pulsed.laser_pulses.data)
        self.assertAlmostEqual(sig[0], 75.42700000000002)
        self.assertAlmostEqual(err[0], 0.22395482225608515)

        self.assertAlmostEqual(odmr.pulsed.timetrace.data[0][0], 6.0)

        self.assertIn("bin width (s)", odmr.pulsed.laser_pulses.params)
        self.assertIn("bin width (s)", odmr.pulsed.timetrace.params)
        self.assertIn("Approx. measurement time (s)", odmr.pulsed.measurement.params)

        self.assertEqual(odmr.get_param_from_filename(unit="dBm"), 22.0)
