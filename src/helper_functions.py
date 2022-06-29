def decibelm_to_watts(dbm_value: float) -> float:
    """ Convert dBm to Watts.  1 W = 10^((1 dBm - 30) / 10) """
    return 10 ** ((dbm_value - 30) / 10)


def format_exponent_as_str(number_to_format: float, decimals: int = 2) -> str:
    """
    Format an exponent as a LaTeX string
    e.g. 0.0001 will be formatted as $1.0 \times 10^{-4}$
    """
    count = 0

    if number_to_format > 1:
        while number_to_format >= 10:
            number_to_format /= 10
            count += 1
    else:
        while number_to_format < 1:
            number_to_format *= 10
            count -= 1

    formatted_str = r"${{ {} }} \times 10^{{ {} }}$".format(round(number_to_format, decimals), count)
    return formatted_str
