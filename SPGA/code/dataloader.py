
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp

ALL_TRAIN = False


class SnoRNADiseaseDataset(Dataset):
    def __init__(self, file_no: str, path: str | Path = "../cv5") -> None:
        self.file_no = str(file_no)
        self.path = Path(path)

        print(f"using {self.path} dataset")
        print(f"loading [{self.path}]")

        self.n_snoRNA: int = 471
        self.m_disease: int = 84

        train_df = self._read_fold_csv(self.path / f"train_fold_{self.file_no}.txt")
        test_df  = self._read_fold_csv(self.path / f"test_fold_{self.file_no}.txt")


        if {"snoRNA_id", "disease_id"}.issubset(train_df.columns):
            self.pos_train = train_df[["snoRNA_id", "disease_id"]].to_numpy()
        else:
            self.pos_train = train_df.iloc[:, :2].to_numpy()

        if {"snoRNA_id", "disease_id"}.issubset(test_df.columns):
            self.pos_test = test_df[["snoRNA_id", "disease_id"]].to_numpy()
        else:
            self.pos_test = test_df.iloc[:, :2].to_numpy()

        self.trainUniqueSnoRNAs = pd.unique(train_df["snoRNA_id"])
        self.testUniqueSnoRNAs  = pd.unique(test_df["snoRNA_id"])

        self.trainSnoRNA = train_df["snoRNA_id"].to_numpy()
        self.trainDisease = train_df["disease_id"].to_numpy()
        self.testSnoRNA = test_df["snoRNA_id"].to_numpy()
        self.testDisease = test_df["disease_id"].to_numpy()

        self.trainSize = len(train_df)
        self.testSize  = len(test_df)

        if ALL_TRAIN:
            self.trainSnoRNA = np.concatenate([self.trainSnoRNA, self.testSnoRNA], axis=0)
            self.trainDisease = np.concatenate([self.trainDisease, self.testDisease], axis=0)
            self.trainSize = self.trainSize + self.testSize

        self.train_bipartite: csr_matrix = self._build_bipartite(self.trainSnoRNA, self.trainDisease)
        self.test_bipartite:  csr_matrix = self._build_bipartite(self.testSnoRNA,  self.testDisease)

        self.SnoRNADiseaseNet  = self.train_bipartite
        self.SnoRNADiseaseNet2 = self.test_bipartite

        print(f"{self.trainSize} interactions for training")
        print(f"{self.testSize} interactions for testing")
        print(f"Sparsity : {(self.trainSize + self.testSize) / self.n_snoRNA / self.m_disease}")

        self.train_pos_diseases: List[np.ndarray] = self._list_positive_diseases(self.train_bipartite)
        self.test_pos_diseases:  List[np.ndarray] = self._list_positive_diseases(self.test_bipartite)

        self.allPos  = self.train_pos_diseases
        self.allPos2 = self.test_pos_diseases

        self.testDict: Dict[int, set] = self._build_test_index()

        print("Ready to go")

    def save_interaction_matrices(self) -> None:
        sp.save_npz(str(self.path / 'train_mat.npz'), self.train_bipartite)
        sp.save_npz(str(self.path / 'test_mat.npz'),  self._build_bipartite(self.testSnoRNA, self.testDisease))

    # 兼容旧名称
    def saveRatingMatrix(self) -> None:
        self.save_interaction_matrices()

    def get_feedback(self, snoRNAs: Sequence[int], diseases: Sequence[int]) -> np.ndarray:

        return np.asarray(self.train_bipartite[snoRNAs, diseases]).astype('uint8').reshape((-1,))


    def getSnoRNADiseaseFeedback(self, snoRNAs, diseases):
        return self.get_feedback(snoRNAs, diseases)

    def get_positive_diseases_train(self, sno_list: Sequence[int]) -> List[np.ndarray]:

        return [self.train_bipartite[s].nonzero()[1] for s in sno_list]

    def get_positive_diseases_test(self, sno_list: Sequence[int]) -> List[np.ndarray]:

        return [self.test_bipartite[s].nonzero()[1] for s in sno_list]


    def getSnoRNAPosDiseases(self, snoRNAs):
        return self.get_positive_diseases_train(snoRNAs)

    def getSnoRNAPosDiseases2(self, snoRNAs):
        return self.get_positive_diseases_test(snoRNAs)


    @staticmethod
    def _read_fold_csv(path: Path) -> pd.DataFrame:

        return pd.read_csv(path)

    def _build_bipartite(self, sno: np.ndarray, dis: np.ndarray) -> csr_matrix:

        data = np.ones(len(sno), dtype=np.float32)
        return csr_matrix((data, (sno, dis)), shape=(self.n_snoRNA, self.m_disease))

    def _list_positive_diseases(self, mat: csr_matrix) -> List[np.ndarray]:

        return [mat[i].nonzero()[1] for i in range(self.n_snoRNA)]

    def _build_test_index(self) -> Dict[int, set]:

        result: Dict[int, set] = defaultdict(set)
        for s, d in zip(self.testSnoRNA, self.testDisease):
            result[s].add(int(d))
        return result


GetData = SnoRNADiseaseDataset



