from math import floor, log10

def FormatToError(mean, lo, hi):
    """
    ...     
    """

    dlo = mean - lo
    dhi = hi - mean

    significant_order = min(floor(log10(abs(dlo))), floor(log10(abs(dhi))))

    exponent = 10**(-significant_order)

    rounded_mean = int(round(exponent*mean))
    rounded_dlo = int(round(exponent*dlo))
    rounded_dhi = int(round(exponent*dhi))
    repre = f'$({rounded_mean:d}^{{ +{rounded_dhi}}}_{{-{rounded_dlo} }}) \cdot 10^{{ {significant_order:d} }}$'
    return repre


    # digits = (n-1)-significant_order
    # rounded_lo = round(lo, digits)
    # rounded_hi = round(hi, digits)
    # rounded_mean = round(mean, digits)
    # lo_repre = int(round((10**digits)*rounded_lo))
    # hi_repre = int(round((10**digits)*rounded_hi))
        
    # if significant_order > 0:
    #     format_string = "{:d}"
    #     rounded_mean = int(rounded_mean)
    #     if n <= significant_order:
    #         lo_repre = f"{int(lo_repre):d}"
    #         hi_repre = f"{int(hi_repre):d}"
    #     else:
    #         std_format_string = f"{{:.{n - significant_order - 1:d}f}}"
    #         lo_repre = std_format_string.format(rounded_lo)
    #         hi_repre = std_format_string.format(rounded_hi)            
    # else:
    #     format_string = f"{{:.{digits:d}f}}"
    # mean_repre = format_string.format(rounded_mean)
    # # return f"{mean_repre}({std_repre})"
    # return f'{mean_repre} \substack {{+{hi_repre} \\ +{lo_repre}}}'