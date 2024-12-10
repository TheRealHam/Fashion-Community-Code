# -----------------------------------------------------------------
# * Binary Classifier
# -----------------------------------------------------------------
# * INPUT: Image (size: 100*100)
# * OUTPUT: Clothe Score
# ? USAGE:
#   - Filter and clean the dataset.
#   - Identify whether a user-uploaded image meets the standard for a clothe image.
# ! IMPORTANT: Ensure robust handling of non-clothing images during preprocessing.
# -----------------------------------------------------------------


import torch
import torch.nn as nn

class BinaryClassifierModel(nn.Module):
    """Binary classifier model definition."""
    def __init__(self):
        super(BinaryClassifierModel, self).__init__()
        # TODO: Ensure model's structure is same with .pth
        self.fc = nn.Linear(512, 1)  # Example fully connected layer

    def forward(self, x):
        """Forward pass."""
        return torch.sigmoid(self.fc(x))  # Example sigmoid for binary output