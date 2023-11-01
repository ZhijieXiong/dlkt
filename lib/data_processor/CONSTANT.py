def datasets_treatable(datasets_merged=None):
    result = ["assist2009", "assist2009-new", "assist2012", "assist2015", "assist2017", "edi2020-task1", "edi2022",
              "SLP-bio", "SLP-mat", "edi2020-task4", "slepemapy", "statics2011", "ednet-kt1"]
    if datasets_merged is None:
        return result
    else:
        return result.extend(datasets_merged)
