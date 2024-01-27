from copy import deepcopy


def datasets_useful_cols(datasets_merged=None):
    result = {
        "assist2009": ["order_id", "user_id", "problem_id", "correct", "skill_id", "school_id", "skill_name"],
        "assist2009-full": ["order_id", "user_id", "problem_id", "correct", "list_skill_ids"],
        "assist2012": ["problem_id", "user_id", "end_time", "correct", "skill_id", "overlap_time", "school_id", "skill"],
        "assist2017": ["studentId", "MiddleSchoolId", "problemId", "skill", "timeTaken", "startTime", "correct"],
        "slepemapy": ["user", "item_asked", "item_answered", "context_name", "type", "time", "response_time",
                      "ip_country", "locations_asked"],
        "statics2011": ["Anon Student Id", "Problem Hierarchy", "Problem Name", "Step Name", "First Attempt", "First Transaction Time"]
    }
    algebra2005 = ["Anon Student Id", "Problem Name", "Step Name", "First Transaction Time", "Correct First Attempt"]
    result["algebra2005"] = deepcopy(algebra2005)
    result["algebra2005"].append("KC(Default)")

    result["algebra2006"] = deepcopy(algebra2005)
    result["algebra2006"].append("KC(Default)")

    result["algebra2008"] = deepcopy(algebra2005)
    result["algebra2008"].append("KC(SubSkills)")

    result["bridge2algebra2006"] = deepcopy(algebra2005)
    result["bridge2algebra2006"].append("KC(SubSkills)")

    result["bridge2algebra2008"] = deepcopy(algebra2005)
    result["bridge2algebra2008"].append("KC(SubSkills)")

    if datasets_merged is None:
        return result
    for k, v in datasets_merged.items():
        result.setdefault(k, v)


def datasets_renamed(datasets_merged=None):
    result = {
        "assist2009": {
            "problem_id": "question_id",
            "skill_id": "concept_id",
            "skill_name": "concept_name"
        },
        "assist2009-full": {
            "problem_id": "question_id",
            "list_skill_ids": "concept_id"
        },
        "assist2012": {
            "problem_id": "question_id",
            "skill_id": "concept_id",
            "end_time": "timestamp",
            "overlap_time": "use_time",
            "skill": "concept_name"
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
            "Problem Hierarchy": "concept_id",
            "First Transaction Time": "timestamp",
            "First Attempt": "correct"
        },
        "ednet-kt1": {
            "tags": "concept_id",
            "elapsed_time": "use_time"
        },
        "slepemapy": {
            "user": "user_id",
            "time": "timestamp",
            "response_time": "use_time",
            "type": "question_type",
            "ip_country": "country_id",
            "context_name": "concept_id"
        }
    }
    algebra2005 = {
        "Anon Student Id": "user_id",
        "Correct First Attempt": "correct",
        "First Transaction Time": "timestamp"
    }
    result["algebra2005"] = deepcopy(algebra2005)
    result["algebra2005"]["KC(Default)"] = "concept_id"

    result["algebra2006"] = deepcopy(algebra2005)
    result["algebra2006"]["KC(Default)"] = "concept_id"

    result["algebra2008"] = deepcopy(algebra2005)
    result["algebra2008"]["KC(SubSkills)"] = "concept_id"

    result["bridge2algebra2006"] = deepcopy(algebra2005)
    result["bridge2algebra2006"]["KC(SubSkills)"] = "concept_id"

    result["bridge2algebra2008"] = deepcopy(algebra2005)
    result["bridge2algebra2008"]["KC(SubSkills)"] = "concept_id"

    if datasets_merged is None:
        return result
    for k, v in datasets_merged.items():
        result.setdefault(k, v)


def datasets_seq_keys(datasets_merged=None):
    result = {
        "assist2009": ["question_seq", "concept_seq", "correct_seq"],
        "assist2009-full": ["question_seq", "concept_seq", "correct_seq"],
        "assist2012": ["question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq"],
        "assist2015": ["question_seq", "correct_seq"],
        "assist2017": ["question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq"],
        "edi2020-task1": ["question_seq", "concept_seq", "correct_seq", "time_seq", "age_seq"],
        "edi2020-task34": ["question_seq", "concept_seq", "correct_seq", "time_seq", "age_seq"],
        "SLP": ["question_seq", "concept_seq", "correct_seq", "interaction_type_seq"],
        "slepemapy": ["question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq"],
        "statics2011": ["question_seq", "concept_seq", "correct_seq", "time_seq"],
        "ednet-kt1": ["question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq"],
        "algebra2005": ["question_seq", "concept_seq", "correct_seq", "time_seq"]
    }
    result["algebra2006"] = result["algebra2005"]
    result["algebra2008"] = result["algebra2005"]
    result["bridge2algebra2006"] = result["algebra2005"]
    result["bridge2algebra2008"] = result["algebra2005"]
    if datasets_merged is None:
        return result
    for k, v in datasets_merged.items():
        result.setdefault(k, v)
