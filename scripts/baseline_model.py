import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import padelpy as pdp

class MoleculeModel:
    def calc_descriptors_rdkit(self):
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        self.header = calc.GetDescriptorNames()
        self.descriptors = dict(zip(self.header, calc.CalcDescriptors(self.mol)))

    def calc_descriptors_padel(self):
        self.padel_descriptors = pdp.from_smiles(self.smiles)

    def calc_fingerprint(self, count=True, radius=2, bits=64):
        if count:
            self.fingerprint = AllChem.GetMorganFingerprint(self.mol, radius)
        else:
            self.fingerprint = AllChem.GetMorganFingerprintAsBitVect(self.mol, radius, nBits=bits)

    def __init__(self, smiles, count=True, radius=2, bits=64):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.calc_descriptors_padel()
        self.calc_descriptors_rdkit()
        self.calc_fingerprint(count, radius, bits)

    def get_mol_representation(self):
        return {'descriptors_rdkit': self.descriptors,
                'descriptors_padel': self.padel_descriptors,
                'fingerprint': self.fingerprint}

    def get_mol_representation_as_series(self):
        return pd.Series(list(self.descriptors.values()) + self.fingerprint)
