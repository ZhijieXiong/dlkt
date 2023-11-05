def datasets_multi_concept(datasets_merged=None):
    result = ["assist2009", "assist2009-new", "assist2012", "ednet-kt1"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_concept(datasets_merged=None):
    result = ["assist2009", "assist2009-new", "assist2012", "assist2017", "SLP-bio", "SLP-mat", "edi2020-task12",
              "edi2020-task1", "edi2020-task4", "slepemapy", "ednet-kt1"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_q_table(datasets_merged=None):
    result = ["assist2009", "assist2009-new", "assist2012", "assist2017", "SLP-bio", "SLP-mat", "edi2020-task12",
              "edi2020-task1", "edi2020-task4", "slepemapy", "ednet-kt1"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_school(datasets_merged=None):
    result = ["assist2009", "assist2009-new", "assist2012", "assist2017", "SLP-bio", "SLP-mat"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_gender(datasets_merged=None):
    result = ["SLP-bio", "SLP-mat", "edi2020-task12", "edi2020-task1", "edi2020-task4"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_age(datasets_merged=None):
    result = ["edi2020-task12", "edi2020-task1", "edi2020-task4"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_campus(datasets_merged=None):
    result = ["SLP-bio", "SLP-mat"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_has_country(datasets_merged=None):
    result = ["slepemapy"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)

