import matplotlib.pyplot as plt
import numpy as np

# Data
x_labels = [
    'Baseline', 'Untargeted FGSM (ε=0.06)', 'Untargeted FGSM (ε=0.08)', 'Untargeted FGSM (ε=0.1)', 
    'Untargeted FGSM Improved (ε=0.06)', 'Untargeted FGSM Improved (ε=0.08)', 'Untargeted FGSM Improved (ε=0.1)', 
    'Targeted FGSM (ε=0.06)', 'Targeted FGSM (ε=0.08)', 'Targeted FGSM (ε=0.1)', 
    'Targeted FGSM Improved (ε=0.08)'
]
training_acc = [0.88670, 0.89569, 0.73177, 0.66954, 0.75406, 0.68298, 0.66954, 0.32183, 0.27136, 0.22678, 0.27136]
validation_acc = [0.83004, 0.88538, 0.70817, 0.65744, 0.73650, 0.64954, 0.57839, 0.32016, 0.28854, 0.25296, 0.28854]

# Bar chart settings
x = np.arange(len(x_labels))
bar_width = 0.4

# Plot
plt.figure(figsize=(12, 6))
plt.bar(x - bar_width / 2, training_acc, bar_width, label='Training Accuracy')
plt.bar(x + bar_width / 2, validation_acc, bar_width, label='Validation Accuracy')

# Customize
plt.xticks(x, x_labels, rotation=45, ha='right', fontsize=10)
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracies for Different FGSM Settings')
plt.legend()
plt.tight_layout()

# Show
plt.show()
