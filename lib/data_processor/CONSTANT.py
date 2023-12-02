def datasets_treatable(datasets_merged=None):
    result = ["assist2009", "assist2009-new", "assist2012", "assist2015", "assist2017", "edi2020-task1", "edi2020-task2"
              , "edi2020-task34",  "SLP-bio", "SLP-mat", "slepemapy", "statics2011", "ednet-kt1", "xes3g5m"]
    if datasets_merged is None:
        return result
    return result.extend(datasets_merged)


def datasets_useful_cols(datasets_merged=None):
    result = {
        "assist2009": ["order_id", "user_id", "problem_id", "correct", "skill_id", "school_id"],
        "assist2012": ["problem_id", "user_id", "end_time", "correct", "skill_id", "overlap_time", "school_id"],
        "assist2017": ["studentId", "MiddleSchoolId", "problemId", "skill", "timeTaken", "startTime", "correct"],
        "slepemapy": ["user", "item_asked", "item_answered", "context_name", "type", "time", "response_time",
                      "ip_country"],
        "statics2011": ["Anon Student Id", "Problem Name", "Step Name", "First Attempt", "First Transaction Time"]
    }
    if datasets_merged is None:
        return result
    for k, v in datasets_merged.items():
        result.setdefault(k, v)


def datasets_renamed(datasets_merged=None):
    result = {
        "assist2009": {
            "problem_id": "question_id",
            "skill_id": "concept_id"
        },
        "assist2012": {
            "problem_id": "question_id",
            "skill_id": "concept_id",
            "end_time": "timestamp",
            "overlap_time": "use_time"
        },
        "assist2015": {
            "sequence_id": "question_id"
        },
        "assist2017": {
            "problemId": "question_id",
            "skill": "concept_id",
            "studentId": "user_id",
            "MiddleSchoolId": "school_id",
            "timeTaken": "use_time",
            "startTime": "timestamp"
        },
        "SLP": {
            "student_id": "user_id",
            "concept": "concept_id",
            "time_access": "timestamp"
        },
        "statics2011": {
            "Anon Student Id": "user_id",
            "First Transaction Time": "timestamp",
            "First Attempt": "answer"
        },
        "ednet-kt1": {
            "tags": "concept_id",
            "elapsed_time": "use_time"
        }
    }
    if datasets_merged is None:
        return result
    for k, v in datasets_merged.items():
        result.setdefault(k, v)


def datasets_seq_keys(datasets_merged=None):
    result = {
        "assist2009": ["question_seq", "concept_seq", "correct_seq"],
        "assist2012": ["question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq"],
        "assist2015": ["question_seq", "correct_seq"],
        "assist2017": ["question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq"],
        "edi2020-task1": ["question_seq", "concept_seq", "correct_seq", "time_seq", "age_seq"],
        "edi2020-task2": ["question_seq", "concept_seq", "correct_seq", "time_seq"],
        "edi2020-task34": ["question_seq", "concept_seq", "correct_seq", "time_seq", "age_seq"],
        "SLP-bio": ["question_seq", "concept_seq", "correct_seq"],
        "SLP-mat": ["question_seq", "concept_seq", "correct_seq"],
        "slepemapy": ["question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq"],
        "statics2011": ["question_seq", "correct_seq", "time_seq"],
        "ednet-kt1": ["question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq"]
    }
    if datasets_merged is None:
        return result
    for k, v in datasets_merged.items():
        result.setdefault(k, v)