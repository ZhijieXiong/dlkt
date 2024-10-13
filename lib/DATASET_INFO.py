def datasets_multi_concept(datasets_merged=None):
    result = ["assist2009", "assist2009-full", "ednet-kt1", "xes3g5m"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_concept(datasets_merged=None):
    result = ["assist2009", "assist2009-full", "assist2012", "assist2017", "SLP-bio", "SLP-mat",
              "edi2020-task1", "edi2020-task34", "slepemapy-anatomy", "ednet-kt1", "xes3g5m"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_q_table(datasets_merged=None):
    result = ["assist2009", "assist2009-full", "assist2012", "assist2017", "SLP-bio", "SLP-mat",
              "edi2020-task1", "edi2020-task34", "slepemapy-anatomy", "ednet-kt1", "xes3g5m"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_school(datasets_merged=None):
    result = ["assist2009", "assist2009-full", "assist2012", "assist2017", "SLP-bio", "SLP-mat"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_gender(datasets_merged=None):
    result = ["SLP-bio", "SLP-mat", "edi2020-task1", "edi2020-task34"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_age(datasets_merged=None):
    result = ["edi2020-task1", "edi2020-task34"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_campus(datasets_merged=None):
    result = ["SLP-bio", "SLP-mat"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_country(datasets_merged=None):
    result = ["slepemapy-anatomy"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)

