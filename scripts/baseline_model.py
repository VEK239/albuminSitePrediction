import json

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import padelpy as pdp

with open('../data/qikprop/qikprop.json') as ouf:
    qikprop_dict = json.loads(ouf.read())

class MoleculeModel:
    def calc_descriptors_rdkit(self):
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        self.header = calc.GetDescriptorNames()
        self.rdkit_descriptors = dict(zip(self.header, calc.CalcDescriptors(self.mol)))

    def calc_descriptors_qikprop(self):
        try:
            self.qikprop_descriptors = qikprop_dict[self.smiles]
        except Exception as e:
            self.qikprop_descriptors = None

    def calc_descriptors_padel(self):
        try:
            self.padel_descriptors = pdp.from_smiles(self.smiles)
        except Exception as e:
            self.padel_descriptors = None

    def calc_fingerprint(self, radius=2, bits=64):
        self.fingerprint = AllChem.GetMorganFingerprintAsBitVect(self.mol, radius, nBits=bits)

    def __init__(self, smiles, radius=2, bits=64, padel=False, rdkit=True, qikprop=True, fp=True):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.padel_descriptors = None
        self.qikprop_descriptors = None
        self.fingerprint = None
        self.rdkit_descriptors = None
        if padel:
            self.calc_descriptors_padel()
        if rdkit:
            self.calc_descriptors_rdkit()
        if qikprop:
            self.calc_descriptors_qikprop()
        if fp:
            self.calc_fingerprint(radius, bits)

    def get_mol_representation(self):
        return {'descriptors_rdkit': self.rdkit_descriptors,
                'descriptors_padel': self.padel_descriptors,
                'descriptors_qikprop': self.qikprop_descriptors,
                'fingerprint': self.fingerprint}

    def get_mol_representation_as_series(self):
        return pd.Series(list(self.rdkit_descriptors.values()) + self.fingerprint)
