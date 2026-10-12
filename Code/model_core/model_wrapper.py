import torch
import torch.nn as nn
from chemprop.models import MoleculeModel
from chemprop.args import TrainArgs

class DMPNNPredictor(nn.Module):
    def __init__(self, args: TrainArgs):
        super(DMPNNPredictor, self).__init__()
        self.model = MoleculeModel(args)
        
    def forward(self, batch, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None):
        # Chemprop's MoleculeModel forward expects a batch of MoleculeGraph objects
        # features_batch etc are optional
        return self.model(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))

def create_model(args: TrainArgs) -> DMPNNPredictor:
    """
    Creates a DMPNN model based on arguments
    """
    return DMPNNPredictor(args)
