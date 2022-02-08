import src.fit_logic as fitlogic
import pandas as pd

f = fitlogic.FitLogic()


def perform_fit(x, y, fit_function: str, estimator: str = "generic"):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    fit = {'1d': {'default': {'fit_function': fit_function, 'estimator': estimator}}}

    user_fit = f.validate_load_fits(fit)

    use_settings = {}
    for key in user_fit["1d"]["default"]["parameters"].keys():
        use_settings[key] = False
    user_fit["1d"]["default"]["use_settings"] = use_settings

    fc = f.make_fit_container("test", "1d")
    fc.set_fit_functions(user_fit["1d"])
    fc.set_current_fit("default")
    fit_x, fit_y, result = fc.do_fit(x, y)
    return fit_x, fit_y, result


def get_fits(dim: str = "1d"):
    return f.fit_list[dim].keys()
