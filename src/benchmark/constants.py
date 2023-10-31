
SMOKING_COLUMNS_TO_SCALE = {
    'smoking':[
        "height(cm)",
        "weight(kg)",
        "waist(cm)",
        "eyesight(left)",
        "eyesight(right)",
        "systolic",
        "relaxation",
        "fasting blood sugar",
        "Cholesterol",
        "triglyceride",
        "HDL",
        "LDL",
        "hemoglobin",
        "serum creatinine",
        "AST",
        "ALT",
        "Gtp",
    ],
    'heart' : [
        "age", "trestbps", "chol", "thalach", "oldpeak",
    ],
    'lumpy' : [
        'x', 'y', 'cld', 'dtr', 'frs', 'pet', 'pre','tmn',
        'tmp','tmx','vap','wet','elevation','X5_Ct_2010_Da','X5_Bf_2010_Da'
    ],
    'machine' : [
        'Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]'
    ],
    'insurance' : [
        'age', 'bmi'
    ]
}

PATHS = {
    'smoking' : 'data/smoking/smoking.csv',
    'heart' : 'data/heart_disease/heart.csv',
    'lumpy' : 'data/lumpy_skin/lumpy_skin.csv',
    'machine' : 'data/machine_maintenance/predictive_maintenance.csv',
    'femnist' : 'data/leaf/data/femnist',
    'synthetic' : 'data/leaf/data/synthetic',
    'insurance' : 'data/health_insurance/insurance.csv'
}