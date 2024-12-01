import matplotlib.pyplot as plt
import numpy as np

# Data for the models
x_labels = ['Baseline', 'Untargeted FGSM (ε=0.06)', 'Untargeted FGSM (ε=0.08)', 'Untargeted FGSM (ε=0.1)', 
            'Untargeted FGSM Improved (ε=0.08)', 'Targeted FGSM (ε=0.06)', 'Targeted FGSM (ε=0.08)', 
            'Targeted FGSM (ε=0.1)', 'Targeted FGSM Improved (ε=0.08)']

# Training and validation accuracies
training_acc = [0.88670, 0.89569, 0.73177, 0.66954, 0.68298, 0.32183, 0.27136, 0.22678, 0.27136]
validation_acc = [0.83004, 0.88538, 0.70817, 0.65744, 0.64954, 0.32016, 0.28854, 0.25296, 0.28854]

# Set the width of the bars
bar_width = 0.35
index = np.arange(len(x_labels))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plots for training and validation accuracies
bars1 = ax.bar(index - bar_width / 2, training_acc, bar_width, label='Training Accuracy')
bars2 = ax.bar(index + bar_width / 2, validation_acc, bar_width, label='Validation Accuracy')

# Labels and title
ax.set_xlabel('Attack Models')
ax.set_ylabel('Accuracy')
ax.set_title('Model Evaluation: Comparison of Training and Validation Accuracies')
ax.set_xticks(index)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()

# # Baseline Model (Finetuned Yolov8) Evaluation
# baseline_training_acc = 0.88670
# baseline_validation_acc = 0.83004

# ## Attack Model (Untargeted FGSM) Evaluation
# # epsilon = 0.06, num_steps = 1
# untargeted_fgsm_training_acc = 0.89569
# untargeted_fgsm_validation_acc = 0.88538
# # epsilon = 0.08, num_steps = 1
# untargeted_fgsm_training_acc2 = 0.73177
# untargeted_fgsm_validation_acc2 = 0.70817
# # epsilon = 0.1, num_steps = 1
# untargeted_fgsm_training_acc2 = 0.66954
# untargeted_fgsm_validation_acc2 = 0.65744

# ## Attack Model (Improved Untargeted FGSM) Evaluation
# # epsilon = 0.08, num_steps = 1, decay_factor = 0.9
# untargeted_fgsm_improved_training_acc = 0.68298
# untargeted_fgsm_improved_validation_acc = 0.64954

# ## Attack Model (Targeted FGSM) Evaluation
# # epsilon = 0.06, num_steps = 1
# targeted_fgsm_training_acc = 0.32183
# targeted_fgsm_validation_acc = 0.32016
# # epsilon = 0.08, num_steps = 1
# targeted_fgsm_training_acc2 = 0.27136
# targeted_fgsm_validation_acc2 = 0.28854
# # epsilon = 0.1, num_steps = 1
# targeted_fgsm_training_acc3 = 0.22678
# targeted_fgsm_validation_acc3 = 0.25296

# ## Attack Model (Improved Targeted FGSM) Evaluation
# # epsilon = 0.08, num_steps = 1, decay_factor = 0.9
# targeted_fgsm_improved_training_acc = 0.27136
# targeted_fgsm_improved_validation_acc = 0.28854