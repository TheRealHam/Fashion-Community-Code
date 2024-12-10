# -----------------------------------------------------------------
# * Style Classifier
# ------------------
# * INPUT: Image (size: 100*100)
# * OUTPUT: Label Temperatures
# ? USAGE:
#   - Categorize clothing images based on style (e.g., formal, casual, sporty).
# TODO: *Expand the label categories to include emerging trends in fashion (able to add new class).
# -----------------------------------------------------------------


import torch
import torch.nn as nn

class StyleClassifierModel(nn.Module):
    """Style classifier model definition."""
    def __init__(self):
        super(StyleClassifierModel, self).__init__()
        # TODO: Ensure model's structure is same with .pth
        self.fc = nn.Linear(512, 50)  # Example fully connected layer

    def forward(self, x):
        """Forward pass."""
        return torch.sigmoid(self.fc(x))  # Example sigmoid for binary output