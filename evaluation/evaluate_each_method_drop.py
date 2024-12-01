import pandas as pd

# Baseline data
baseline_data = {
    "training": {
        "Hydrant": 0.99666, "Car": 0.87785, "Bridge": 0.92484, "Other": 0.74480,
        "Motorcycle": 0.51852, "Bus": 0.94372, "Chimney": 0.61538, "Bicycle": 0.91873,
        "Mountain": 0.09091, "Crosswalk": 0.93170, "Traffic Light": 0.88060, "Palm": 0.89044
    },
    "evaluation": {
        "Hydrant": 1.00000, "Car": 0.78378, "Bridge": 0.83784, "Other": 0.64865,
        "Motorcycle": 0.68919, "Bus": 0.97297, "Chimney": 0.64706, "Bicycle": 0.89189,
        "Mountain": 0.00000, "Crosswalk": 0.89189, "Traffic Light": 0.86486, "Palm": 0.78378
    },
}

data = {
    "untargeted_fgsm": {
        "epsilon_0.06": {
            "training": {
                "Hydrant": 0.98998, "Car": 0.90411, "Bridge": 0.92484, "Other": 0.75627,
                "Motorcycle": 0.66667, "Bus": 0.98701, "Chimney": 0.97436, "Bicycle": 0.77961,
                "Mountain": 0.00000, "Crosswalk": 0.82715, "Traffic Light": 0.97151, "Palm": 0.99301
            },
            "evaluation": {
                "Hydrant": 1.00000, "Car": 0.89189, "Bridge": 0.94595, "Other": 0.70270,
                "Motorcycle": 0.82432, "Bus": 1.00000, "Chimney": 0.82353, "Bicycle": 0.79730,
                "Mountain": 0.00000, "Crosswalk": 0.78378, "Traffic Light": 0.95946, "Palm": 0.98649
            },
        },
        "epsilon_0.08": {
            "training": {
                "Hydrant": 0.92817, "Car": 0.71047, "Bridge": 0.66493, "Other": 0.53871,
                "Motorcycle": 0.38889, "Bus": 0.95065, "Chimney": 0.93590, "Bicycle": 0.41185,
                "Mountain": 0.00000, "Crosswalk": 0.58347, "Traffic Light": 0.92605, "Palm": 0.98951
            },
            "evaluation": {
                "Hydrant": 0.93243, "Car": 0.59459, "Bridge": 0.72297, "Other": 0.54054,
                "Motorcycle": 0.56081, "Bus": 0.96622, "Chimney": 0.61765, "Bicycle": 0.39865,
                "Mountain": 0.00000, "Crosswalk": 0.47973, "Traffic Light": 0.93919, "Palm": 0.98649
            },
        },
        "epsilon_0.1": {
            "training": {
                "Hydrant": 0.89310, "Car": 0.62700, "Bridge": 0.54280, "Other": 0.48459,
                "Motorcycle": 0.29630, "Bus": 0.93247, "Chimney": 0.89744, "Bicycle": 0.29063,
                "Mountain": 0.00000, "Crosswalk": 0.50337, "Traffic Light": 0.90231, "Palm": 0.98718
            },
            "evaluation": {
                "Hydrant": 0.90541, "Car": 0.54054, "Bridge": 0.64865, "Other": 0.47297,
                "Motorcycle": 0.45946, "Bus": 0.94595, "Chimney": 0.58824, "Bicycle": 0.28378,
                "Mountain": 0.00000, "Crosswalk": 0.43243, "Traffic Light": 0.93243, "Palm": 0.98649
            },
        },
    },
    "untargeted_fgsm_improved": {
        "epsilon_0.06": {
            "training": {
                "Hydrant": 0.95991, "Car": 0.70148, "Bridge": 0.78288, "Other": 0.51828,
                "Motorcycle": 0.40741, "Bus": 0.89784, "Chimney": 0.61538, "Bicycle": 0.66529,
                "Mountain": 0.00000, "Crosswalk": 0.77066, "Traffic Light": 0.83582, "Palm": 0.93590
            },
            "evaluation": {
                "Hydrant": 1.00000, "Car": 0.58108, "Bridge": 0.79730, "Other": 0.43243,
                "Motorcycle": 0.51351, "Bus": 0.91892, "Chimney": 0.64706, "Bicycle": 0.67568,
                "Mountain": 0.00000, "Crosswalk": 0.74324, "Traffic Light": 0.83784, "Palm": 0.90541
            },
        },
        "epsilon_0.08": {
            "training": {
                "Hydrant": 0.93541, "Car": 0.63984, "Bridge": 0.67223, "Other": 0.44588,
                "Motorcycle": 0.33333, "Bus": 0.87359, "Chimney": 0.61538, "Bicycle": 0.43939,
                "Mountain": 0.00000, "Crosswalk": 0.61551, "Traffic Light": 0.82090, "Palm": 0.93357
            },
            "evaluation": {
                "Hydrant": 0.95946, "Car": 0.47297, "Bridge": 0.70270, "Other": 0.39189,
                "Motorcycle": 0.40541, "Bus": 0.90541, "Chimney": 0.52941, "Bicycle": 0.41892,
                "Mountain": 0.00000, "Crosswalk": 0.51351, "Traffic Light": 0.83784, "Palm": 0.93243
            },
        },
        "epsilon_0.1": {
            "training": {
                "Hydrant": 0.86526, "Car": 0.52540, "Bridge": 0.47599, "Other": 0.38208,
                "Motorcycle": 0.22222, "Bus": 0.83463, "Chimney": 0.56410, "Bicycle": 0.23003,
                "Mountain": 0.00000, "Crosswalk": 0.46374, "Traffic Light": 0.78426, "Palm": 0.92774
            },
            "evaluation": {
                "Hydrant": 0.90541, "Car": 0.41892, "Bridge": 0.58108, "Other": 0.31081,
                "Motorcycle": 0.31081, "Bus": 0.87838, "Chimney": 0.52941, "Bicycle": 0.21622,
                "Mountain": 0.00000, "Crosswalk": 0.41892, "Traffic Light": 0.83784, "Palm": 0.93243
            },
        },
    },
    "targeted_fgsm":{
        "epsilon_0.08":{
            "training": {
                "Hydrant": 0.53229, "Car": 0.12900, "Bridge": 0.14614, "Other": 0.19427,
                "Motorcycle": 0.29630, "Bus": 0.33160, "Chimney": 0.33333, "Bicycle": 0.15014,
                "Mountain": 0.00000, "Crosswalk": 0.27909, "Traffic Light": 0.28765, "Palm": 0.77156
            },
            "evaluation": {
                "Hydrant": 0.56757, "Car": 0.12162, "Bridge": 0.25676, "Other": 0.14865,
                "Motorcycle": 0.14865, "Bus": 0.33784, "Chimney": 0.11765, "Bicycle": 0.10811,
                "Mountain": 0.00000, "Crosswalk": 0.33784, "Traffic Light": 0.20270, "Palm": 0.70270
            },
        },
        "epsilon_0.06": {
            "training": {
                "Hydrant": 0.61136, "Car": 0.20120, "Bridge": 0.25261, "Other": 0.20287,
                "Motorcycle": 0.29630, "Bus": 0.35411, "Chimney": 0.33333, "Bicycle": 0.21625,
                "Mountain": 0.00000, "Crosswalk": 0.34907, "Traffic Light": 0.33514, "Palm": 0.74476
            },
            "evaluation": {
                "Hydrant": 0.56757, "Car": 0.25676, "Bridge": 0.31081, "Other": 0.16216,
                "Motorcycle": 0.16216, "Bus": 0.35135, "Chimney": 0.11765, "Bicycle": 0.17568,
                "Mountain": 0.00000, "Crosswalk": 0.39189, "Traffic Light": 0.24324, "Palm": 0.63514
            },
        },
        "epsilon_0.1": {
            "training": {
                "Hydrant": 0.41759, "Car": 0.08191, "Bridge": 0.06889, "Other": 0.18925,
                "Motorcycle": 0.18519, "Bus": 0.31342, "Chimney": 0.28205, "Bicycle": 0.06612,
                "Mountain": 0.00000, "Crosswalk": 0.22175, "Traffic Light": 0.24966, "Palm": 0.77622
            },
            "evaluation": {
                "Hydrant": 0.48649, "Car": 0.09459, "Bridge": 0.16216, "Other": 0.18919,
                "Motorcycle": 0.08108, "Bus": 0.32432, "Chimney": 0.05882, "Bicycle": 0.01351,
                "Mountain": 0.00000, "Crosswalk": 0.28378, "Traffic Light": 0.21622, "Palm": 0.72973
            },
        }
        
    },
    "targeted_fgsm_improved":{
        "epsilon_0.08":{
            "training": {
                "Hydrant": 0.53229, "Car": 0.12900, "Bridge": 0.14614, "Other": 0.19427,
                "Motorcycle": 0.29630, "Bus": 0.33160, "Chimney": 0.33333, "Bicycle": 0.15014,
                "Mountain": 0.00000, "Crosswalk": 0.27909, "Traffic Light": 0.28765, "Palm": 0.77156
            },
            "evaluation": {
                "Hydrant": 0.56757, "Car": 0.12162, "Bridge": 0.25676, "Other": 0.14865,
                "Motorcycle": 0.14865, "Bus": 0.33784, "Chimney": 0.11765, "Bicycle": 0.10811,
                "Mountain": 0.00000, "Crosswalk": 0.33784, "Traffic Light": 0.20270, "Palm": 0.70270
            },
        }
    }
}

def calculate_drops(data, baseline_data):
    rows = []
    for method, epsilons in data.items():
        for epsilon, datasets in epsilons.items():
            for dataset_type, classes in datasets.items():
                for class_label, accuracy in classes.items():
                    baseline_accuracy = baseline_data[dataset_type][class_label]
                    drop = baseline_accuracy - accuracy
                    rows.append({
                        "Method": method,
                        "Epsilon": epsilon,
                        "Dataset": dataset_type,
                        "Class": class_label,
                        "Baseline Accuracy": baseline_accuracy,
                        "Method Accuracy": accuracy,
                        "Accuracy Drop": drop,
                    })

    return pd.DataFrame(rows)

def calculate_average_drop_per_method_and_epsilon(drop_table):

    avg_drop_per_method = (
        drop_table.groupby(["Method", "Epsilon"])["Accuracy Drop"]
        .mean()
        .reset_index()
        .rename(columns={"Accuracy Drop": "Average Drop"})
        .sort_values("Average Drop", ascending=False)
    )
    return avg_drop_per_method


drop_table = calculate_drops(data, baseline_data)

avg_drop_per_method = calculate_average_drop_per_method_and_epsilon(drop_table)

output_file = "average_drop_per_method.csv"
avg_drop_per_method.to_csv(output_file, index=False)
