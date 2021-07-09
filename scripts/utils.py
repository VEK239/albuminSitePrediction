from collections import defaultdict

import pandas as pd
from tqdm.notebook import tqdm

from scripts.baseline_model import MoleculeModel


def generate_descriptors(data, use_fp=True, fp_size=32, fp_radius=2, descriptors_rdkit=True, descriptors_qikprop=True,
                         descriptors_padel=False):
    features = defaultdict(list)
    for smi in tqdm(data.smiles):
        print(smi)
        mol = MoleculeModel(smi, fp=use_fp, bits=fp_size, radius=fp_radius, padel=descriptors_padel,
                            rdkit=descriptors_rdkit, qikprop=descriptors_qikprop)
        representation = mol.get_mol_representation()

        if use_fp:
            fp = representation['fingerprint']
            for i in fp:
                bit = fp[i]
                features['fp' + str(i)].append(bit)

        if descriptors_rdkit:
            descriptors = representation['descriptors_rdkit']
            if descriptors is not None:
                for k, v in descriptors.items():
                    features[k].append(v)
            else:
                for k in features.keys():
                    features[k].append(None)

        if descriptors_qikprop:
            descriptors = representation['descriptors_qikprop']
            if descriptors is not None:
                for k, v in descriptors.items():
                    features[k].append(v)
            else:
                for k in features.keys():
                    features[k].append(None)

        if descriptors_padel:
            descriptors = representation['descriptors_padel']
            if descriptors is not None:
                for k, v in descriptors.items():
                    features[k].append(v)
            else:
                for k in features.keys():
                    features[k].append(None)

    for k, v in features.items():
        data[k] = pd.Series(v)
    data = data.dropna()
    return data
