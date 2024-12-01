from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from fgsm_untargeted_improved import ImprovedUntargetedFGSM 
from fgsm_untargeted import UntargetedFGSM
from fgsm_targeted import TargetedFGSM
from fgsm_targeted_improved import ImprovedTargetedFGSM

import argparse

CLASS = {0: 'Bicycle', 1: 'Bridge', 2: 'Bus', 3: 'Car', 4: 'Chimney', 5: 'Crosswalk', 
         6: 'Hydrant', 7: 'Motorcycle', 8: 'Mountain', 9: 'Other', 10: 'Palm', 
         11: 'Stairs', 12: 'Traffic Light'}

TARGETED_CLASS = {
    "Bicycle":"Motorcycle", "Bridge":"Mountain","Bus":"Car","Car":"Bus","Chimney":"Mountain",
    "Crosswalk":"Traffic Light","Hydrant":"Traffic Light","Motorcycle":"Bicycle",
    "Mountain":"Chimney","Other":"Palm","Palm":"Mountain","Stairs":"Other","Traffic Light":"Crosswalk"
}

def import_model(model_path, attack_type, epsilon, num_steps, decay_factor):
    if attack_type == "untargeted_fgsm_improved":
        attack_model = ImprovedUntargetedFGSM(model_path, CLASS, epsilon=epsilon, num_steps=num_steps, decay_factor=decay_factor)

    elif attack_type == "untargeted_fgsm":
        attack_model = UntargetedFGSM(model_path, CLASS, epsilon=epsilon, num_steps=num_steps)
    elif attack_type == "targeted_fgsm":
        attack_model = TargetedFGSM(model_path, CLASS, epsilon=epsilon, num_steps=num_steps)
    elif attack_type == "targeted_fgsm_improved":
        attack_model = ImprovedTargetedFGSM(model_path, CLASS, epsilon=epsilon, num_steps=num_steps, decay_factor=decay_factor)
    
    return attack_model

def gen_folder_name(attack_type,model):
    if attack_type == "untargeted_fgsm":
        if model.get_num_steps() == 1:
            return f"untargeted_fgsm-{model.get_epsilon()}"
        else:
            return f"untargeted_fgsm-{model.get_epsilon()}-{model.get_num_steps()}"
    elif attack_type == "untargeted_fgsm_improved":
        if model.get_num_steps() == 1:
            return f"untargeted_fgsm_improved-{model.get_epsilon()}-{model.get_decay_factor()}"
        else:
            return f"untargeted_fgsm_improved-{model.get_epsilon()}-{model.get_num_steps()}-{model.get_decay_factor()}"
    elif attack_type == "targeted_fgsm":
        if model.get_num_steps() == 1:
            return f"targeted_fgsm-{model.get_epsilon()}"
        else:
            return f"targeted_fgsm-{model.get_epsilon()}-{model.get_num_steps()}"
    elif attack_type == "targeted_fgsm_improved":
        if model.get_num_steps() == 1:
            return f"targeted_fgsm_improved-{model.get_epsilon()}-{model.get_decay_factor()}"
        else:
            return f"targeted_fgsm_improved-{model.get_epsilon()}-{model.get_num_steps()}-{model.get_decay_factor()}"
    

if __name__ == "__main__":

    #read attack type from args
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', type=str, default='targeted_fgsm')
    parser.add_argument('--num_steps', type=str, default=1 )
    parser.add_argument('--epsilon', type=str, default=0.1)
    parser.add_argument('--decay_factor', type=str, default=0.9)
    parser.add_argument('--model_path', type=str, default='../models/YOLO_Classification/train4/weights/best.pt')
    parser.add_argument('--root_data_dir', type=str, default='../data')

    args = parser.parse_args()

    root_dir=args.root_data_dir
    model_path=args.model_path
    # change attack_type as needed
    attack_type=args.attack_type
    num_steps=int(args.num_steps)
    epsilon=float(args.epsilon)
    decay_factor=float(args.decay_factor)

    attack_model = import_model(model_path, attack_type, epsilon=epsilon, num_steps=num_steps, decay_factor=decay_factor)
    folder_name = gen_folder_name(attack_type, attack_model)

    for category in ["Training", "Validation"]:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root_dir, category)):
            for dirname in dirnames:
                print(dirname)
                if not os.path.exists(f"generated_data/{folder_name}/"+category+"/"+dirname):
                    os.makedirs(f"generated_data/{folder_name}/"+category+"/"+dirname)
                dir=os.path.join(dirpath, dirname)
                for filename in os.listdir(dir):
                    if filename.endswith(".png"):
                        if attack_type == "untargeted_fgsm" or attack_type == "untargeted_fgsm_improved":
                            perturbed_image_np = attack_model.generate_adversarial_example(os.path.join(dir, filename))
                        else:
                            target_class = TARGETED_CLASS[dirname]
                            perturbed_image_np = attack_model.generate_adversarial_example(os.path.join(dir, filename), target_class)
                        if attack_type == "untargeted_fgsm_improved" or attack_type == "targeted_fgsm_improved":
                            image_name=f"{folder_name}-eps{attack_model.get_epsilon()}-n{attack_model.get_num_steps()}-d{attack_model.get_decay_factor()}-{filename}"
                        elif attack_type == "untargeted_fgsm" or attack_type == "targeted_fgsm":
                            image_name=f"{folder_name}-eps{attack_model.get_epsilon()}-n{attack_model.get_num_steps()}-{filename}"
                        image_name=f"{folder_name}-eps{attack_model.get_epsilon()}-n{attack_model.get_num_steps()}-{filename}"
                        print("Saving image: ", f"generated_data/{folder_name}/{category}/{dirname}/{image_name}")
                        cv2.imwrite(f"generated_data/{folder_name}/{category}/{dirname}/{image_name}", cv2.cvtColor(perturbed_image_np, cv2.COLOR_RGB2BGR))
                        
                       
                        

    

