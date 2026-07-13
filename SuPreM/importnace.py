import joblib

model_path = (
    "/home/s2347484/Seg/SuPreM/results/"
    "train_validation_inference/random_forest_stacking_fixed/"
    "rf_stacking.joblib"
)

feature_names = [
    "clip_bg",
    "clip_pancreas",
    "clip_kidney",
    "clip_liver",
    "segresnet_bg",
    "segresnet_pancreas",
    "segresnet_kidney",
    "segresnet_liver",
    "swin_bg",
    "swin_pancreas",
    "swin_kidney",
    "swin_liver",
]

model = joblib.load(model_path)

ranked_importances = sorted(
    zip(feature_names, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True,
)

for name, importance in ranked_importances:
    print(f"{name}: {importance:.6f}")