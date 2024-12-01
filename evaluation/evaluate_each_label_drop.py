import pandas as pd

data = {
    "Label": [
        "Hydrant", "Car", "Bridge", "Other", "Motorcycle", "Bus", "Chimney", "Bicycle",
        "Mountain", "Crosswalk", "Traffic Light", "Palm"
    ],
    "untargeted_fgsm_0.1": [0.158, 0.144, 0.135, 0.123, 0.119, 0.101, 0.111, 0.098, 0.0, 0.121, 0.145, 0.155],
    "untargeted_fgsm_0.08": [0.127, 0.112, 0.101, 0.089, 0.075, 0.084, 0.097, 0.078, 0.0, 0.097, 0.112, 0.119],
    "untargeted_fgsm_0.06": [-0.045, -0.032, -0.022, -0.018, -0.012, -0.009, -0.011, -0.015, 0.0, -0.014, -0.019, -0.023],
    "untargeted_fgsm_improved_0.1": [0.201, 0.176, 0.167, 0.159, 0.147, 0.131, 0.142, 0.128, 0.0, 0.153, 0.175, 0.185],
    "untargeted_fgsm_improved_0.08": [0.154, 0.131, 0.123, 0.112, 0.097, 0.104, 0.115, 0.098, 0.0, 0.113, 0.132, 0.145],
    "untargeted_fgsm_improved_0.06": [0.093, 0.072, 0.068, 0.057, 0.045, 0.053, 0.061, 0.048, 0.0, 0.058, 0.071, 0.081],
    # "targeted_fgsm_0.08": [0.53229, 0.129, 0.14614, 0.19427, 0.2963, 0.3316, 0.33333, 0.15014, 0.0, 0.27909, 0.28765, 0.77156],
    # "targeted_fgsm_improved_0.08": [0.53229, 0.129, 0.14614, 0.19427, 0.2963, 0.3316, 0.33333, 0.15014, 0.0, 0.27909, 0.28765, 0.77156],
}


df = pd.DataFrame(data)
df["Average Drop (%)"] = df.iloc[:, 1:].mean(axis=1) * 100  

average_drop_table = df[["Label", "Average Drop (%)"]].sort_values("Average Drop (%)", ascending=False)
output_file = "average_drop_per_label.csv"
average_drop_table.to_csv(output_file, index=False)