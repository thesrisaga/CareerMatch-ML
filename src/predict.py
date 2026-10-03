
import pandas as pd
import joblib


MODEL_PATH = "careermatch_model.pkl"

model = joblib.load(MODEL_PATH)


def predict_compatibility(candidate, job):
    """
    Predict candidate-job compatibility.

    Parameters
    ----------
    candidate : dict
        Candidate information.

    job : dict
        Job information.

    Returns
    -------
    probability : float
        Probability of high compatibility.

    prediction : int
        0 = Low Compatibility
        1 = High Compatibility
    """

    skill_pairs = {
        "python": "python_required",
        "sql": "sql_required",
        "machine_learning": "ml_required",
        "deep_learning": "dl_required",
        "power_bi": "power_bi_required",
        "excel": "excel_required",
        "cloud": "cloud_required"
    }

    matched = 0
    required = 0

    for candidate_skill, job_skill in skill_pairs.items():

        if job[job_skill] == 1:
            required += 1

            if candidate[candidate_skill] == 1:
                matched += 1

    skill_match = (
        matched / required * 100
        if required > 0
        else 0
    )

    experience_match = (
        min(
            candidate["experience_years"] /
            job["required_experience"],
            1
        )
        if job["required_experience"] > 0
        else 1
    )

    education_mapping = {
        "Diploma": 1,
        "Bachelor's": 2,
        "Master's": 3
    }

    education_match = int(
        education_mapping[candidate["education_level"]]
        >= education_mapping[job["minimum_education"]]
    )

    project_score = min(
        candidate["project_count"] / 5,
        1
    )

    internship_score = candidate["internship_experience"]

    project_relevance = (
        0.7 * project_score +
        0.3 * internship_score
    )

    input_data = pd.DataFrame([{
        **candidate,
        **job,
        "skill_match_percentage": round(
            skill_match, 2
        ),
        "experience_match": round(
            experience_match, 2
        ),
        "education_match": education_match,
        "project_relevance": round(
            project_relevance, 2
        )
    }])

    feature_columns = [
        "education_level",
        "experience_years",
        "python",
        "sql",
        "machine_learning",
        "deep_learning",
        "power_bi",
        "excel",
        "cloud",
        "certification_count",
        "project_count",
        "internship_experience",
        "job_domain",
        "required_experience",
        "python_required",
        "sql_required",
        "ml_required",
        "dl_required",
        "power_bi_required",
        "excel_required",
        "cloud_required",
        "minimum_education",
        "skill_match_percentage",
        "experience_match",
        "education_match",
        "project_relevance"
    ]

    input_data = input_data[feature_columns]

    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = int(probability >= 0.5)

    return probability, prediction
