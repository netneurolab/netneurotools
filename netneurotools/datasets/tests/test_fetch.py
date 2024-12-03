"""For testing netneurotools.datasets.fetch_* functionality."""

import os
import pytest
from pathlib import Path
import numpy as np
from netneurotools import datasets


@pytest.mark.fetcher
class TestFetchTemplate:
    """Test fetching of template datasets."""

    @pytest.mark.parametrize(
        "version", ["fsaverage", "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6"]
    )
    def test_fetch_fsaverage(self, tmp_path, version):
        """Test fetching of fsaverage surfaces."""
        fsaverage = datasets.fetch_fsaverage(
            version=version, data_dir=tmp_path, verbose=0
        )
        for k in ["orig", "white", "smoothwm", "pial", "inflated", "sphere"]:
            assert k in fsaverage
            assert fsaverage[k].L.exists()
            assert fsaverage[k].R.exists()

    @pytest.mark.parametrize(
        "version", ["fsaverage", "fsaverage4", "fsaverage5", "fsaverage6"]
    )
    def test_fetch_fsaverage_curated(self, tmp_path, version):
        """Test fetching of curated fsaverage surfaces."""
        fsaverage = datasets.fetch_fsaverage_curated(
            version=version, data_dir=tmp_path, verbose=0
        )
        for k in ["white", "pial", "inflated", "sphere", "medial", "sulc", "vaavg"]:
            assert k in fsaverage
            assert fsaverage[k].L.exists()
            assert fsaverage[k].R.exists()

    def test_fetch_hcp_standards(self, tmp_path):
        """Test fetching of HCP standard meshes."""
        hcp = datasets.fetch_hcp_standards(data_dir=tmp_path, verbose=0)
        assert hcp.exists()

    @pytest.mark.parametrize("version", ["fslr4k", "fslr8k", "fslr32k", "fslr164k"])
    def test_fetch_fslr_curated(self, tmp_path, version):
        """Test fetching of curated fsLR surfaces."""
        fslr = datasets.fetch_fslr_curated(
            version=version, data_dir=tmp_path, verbose=0
        )
        for k in [
            "midthickness",
            "inflated",
            "veryinflated",
            "sphere",
            "medial",
            "sulc",
            "vaavg",
        ]:
            if version in ["fslr4k", "fslr8k"] and k == "veryinflated":
                continue
            assert k in fslr
            assert fslr[k].L.exists()
            assert fslr[k].R.exists()

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_fetch_civet(self, tmp_path, version):
        """Test fetching of CIVET templates."""
        civet = datasets.fetch_civet(version=version, data_dir=tmp_path, verbose=0)
        for key in ("mid", "white"):
            assert key in civet
            assert civet[key].L.exists()
            assert civet[key].R.exists()

    @pytest.mark.parametrize("version", ["civet41k", "civet164k"])
    def test_fetch_civet_curated(self, tmp_path, version):
        """Test fetching of curated CIVET templates."""
        civet = datasets.fetch_civet_curated(
            version=version, data_dir=tmp_path, verbose=0
        )
        for k in [
            "white",
            "midthickness",
            "inflated",
            "veryinflated",
            "sphere",
            "medial",
            "sulc",
            "vaavg",
        ]:
            assert k in civet
            assert civet[k].L.exists()
            assert civet[k].R.exists()

    def test_fetch_conte69(self, tmp_path):
        """Test fetching of Conte69 surfaces."""
        conte = datasets.fetch_conte69(data_dir=tmp_path, verbose=0)
        assert all(
            hasattr(conte, k) for k in ["midthickness", "inflated", "vinflated", "info"]
        )

    def test_fetch_yerkes19(self, tmp_path):
        """Test fetching of Yerkes19 surfaces."""
        yerkes19 = datasets.fetch_yerkes19(data_dir=tmp_path, verbose=0)
        assert all(
            hasattr(yerkes19, k) for k in ["midthickness", "inflated", "vinflated"]
        )


@pytest.mark.fetcher
class TestFetchAtlas:
    """Test fetching of atlas datasets."""

    @pytest.mark.parametrize(
        "version, expected",
        [
            ("MNI152NLin2009aSym", [1, 1, 1, 1, 1]),
            ("fsaverage", [2, 2, 2, 2, 2]),
            ("fsaverage5", [2, 2, 2, 2, 2]),
            ("fsaverage6", [2, 2, 2, 2, 2]),
            ("fslr32k", [2, 2, 2, 2, 2]),
            ("gcs", [2, 2, 2, 2, 6]),
        ],
    )
    def test_fetch_cammoun2012(self, tmp_path, version, expected):
        """Test fetching of Cammoun2012 parcellations."""
        keys = ["scale033", "scale060", "scale125", "scale250", "scale500"]
        cammoun = datasets.fetch_cammoun2012(version, data_dir=tmp_path, verbose=0)

        # output has expected keys
        assert all(hasattr(cammoun, k) for k in keys)
        # and keys are expected lengths!
        for k, e in zip(keys, expected):
            out = getattr(cammoun, k)
            if isinstance(out, (tuple, list)):
                assert len(out) == e
            else:
                assert isinstance(out, Path) and str(out).endswith(".nii.gz")

    @pytest.mark.parametrize(
        "version", ["fsaverage", "fsaverage5", "fsaverage6", "fslr32k"]
    )
    def test_fetch_schaefer2018(self, tmp_path, version):
        """Test fetching of Schaefer2018 parcellations."""
        keys = [
            f"{p}Parcels{n}Networks" for p in range(100, 1001, 100) for n in [7, 17]
        ]
        schaefer = datasets.fetch_schaefer2018(version, data_dir=tmp_path, verbose=0)

        if version == "fslr32k":
            assert all(k in schaefer and os.path.isfile(schaefer[k]) for k in keys)
        else:
            for k in keys:
                assert k in schaefer
                assert len(schaefer[k]) == 2
                assert all(os.path.isfile(hemi) for hemi in schaefer[k])

    def test_fetch_mmpall(self, tmp_path):
        """Test fetching of MMPAll parcellations."""
        mmp = datasets.fetch_mmpall(data_dir=tmp_path, verbose=0)
        assert len(mmp) == 2
        assert all(os.path.isfile(hemi) for hemi in mmp)
        assert all(hasattr(mmp, attr) for attr in ("L", "R"))

    def test_fetch_pauli2018(self, tmp_path):
        """Test fetching of Pauli2018 parcellations."""
        pauli = datasets.fetch_pauli2018(data_dir=tmp_path, verbose=0)
        assert all(
            hasattr(pauli, k) and os.path.isfile(pauli[k])
            for k in ["probabilistic", "deterministic", "info"]
        )

    @pytest.mark.xfail
    def test_fetch_tian2020msa(self, tmp_path):
        """Test fetching of tian2020msa parcellations."""
        assert False

    def test_fetch_voneconomo(self, tmp_path):
        """Test fetching of von Economo parcellations."""
        vek = datasets.fetch_voneconomo(data_dir=tmp_path, verbose=0)
        assert all(hasattr(vek, k) and len(vek[k]) == 2 for k in ["gcs", "ctab"])
        assert isinstance(vek.get("info"), Path)


@pytest.mark.fetcher
class TestFetchProject:
    """Test fetching of project datasets."""

    def test_fetch_vazquez_rodriguez2019(self, tmp_path):
        """Test fetching of Vazquez-Rodriguez2019 dataset."""
        vazquez = datasets.fetch_vazquez_rodriguez2019(data_dir=tmp_path, verbose=0)
        for k in ["rsquared", "gradient"]:
            assert hasattr(vazquez, k)
            assert isinstance(getattr(vazquez, k), np.ndarray)

    @pytest.mark.xfail
    def test_fetch_mirchi2018(self, tmp_path):
        """Test fetching of Mirchi2018 dataset."""
        X, Y = datasets.fetch_mirchi2018(data_dir=tmp_path, verbose=0)
        assert isinstance(X, np.ndarray)
        assert X.shape == (73, 198135)
        assert isinstance(Y, np.ndarray)
        assert Y.shape == (73, 13)

    def test_fetch_hansen_manynetworks(self, tmp_path):
        """Test fetching of Hansen et al., 2023 many-networks dataset."""
        hansen = datasets.fetch_hansen_manynetworks(data_dir=tmp_path, verbose=0)
        assert hansen.exists()
        # assert "cammoun033" in hansen
        # assert "gene" in hansen["cammoun033"]
        # assert isinstance(hansen["cammoun033"]["gene"], Path)

    def test_fetch_hansen_receptors(self, tmp_path):
        """Test fetching of Hansen et al., 2022 receptor dataset."""
        hansen = datasets.fetch_hansen_receptors(data_dir=tmp_path, verbose=0)
        assert hansen.exists()

    def test_fetch_hansen_genescognition(self, tmp_path):
        """Test fetching of Hansen et al., 2021 gene-cognition dataset."""
        hansen = datasets.fetch_hansen_genescognition(data_dir=tmp_path, verbose=0)
        assert hansen.exists()

    def test_fetch_hansen_brainstemfc(self, tmp_path):
        """Test fetching of Hansen et al., 2024 brainstem dataset."""
        hansen = datasets.fetch_hansen_brainstemfc(data_dir=tmp_path, verbose=0)
        assert hansen.exists()

    def test_fetch_shafiei_megfmrimapping(self, tmp_path):
        """Test fetching of Shafiei et al., 2022 & 2023 HCP-MEG dataset."""
        shafiei = datasets.fetch_shafiei_megfmrimapping(data_dir=tmp_path, verbose=0)
        assert shafiei.exists()

    def test_fetch_shafiei_megdynamics(self, tmp_path):
        """Test fetching of Shafiei et al., 2022 & 2023 HCP-MEG dataset."""
        shafiei = datasets.fetch_shafiei_megdynamics(data_dir=tmp_path, verbose=0)
        assert shafiei.exists()

    def test_fetch_suarez_mami(self, tmp_path):
        """Test fetching of Suarez et al., 2022 mami dataset."""
        suarez = datasets.fetch_suarez_mami(data_dir=tmp_path, verbose=0)
        assert suarez.exists()

    @pytest.mark.parametrize(
        "dataset, expected",
        [
            ("celegans", ["conn", "dist", "labels", "ref"]),
            ("drosophila", ["conn", "coords", "labels", "networks", "ref"]),
            ("human_func_scale033", ["conn", "coords", "labels", "ref"]),
            ("human_func_scale060", ["conn", "coords", "labels", "ref"]),
            ("human_func_scale125", ["conn", "coords", "labels", "ref"]),
            ("human_func_scale250", ["conn", "coords", "labels", "ref"]),
            ("human_func_scale500", ["conn", "coords", "labels", "ref"]),
            ("human_struct_scale033", ["conn", "coords", "dist", "labels", "ref"]),
            ("human_struct_scale060", ["conn", "coords", "dist", "labels", "ref"]),
            ("human_struct_scale125", ["conn", "coords", "dist", "labels", "ref"]),
            ("human_struct_scale250", ["conn", "coords", "dist", "labels", "ref"]),
            ("human_struct_scale500", ["conn", "coords", "dist", "labels", "ref"]),
            ("macaque_markov", ["conn", "dist", "labels", "ref"]),
            ("macaque_modha", ["conn", "coords", "dist", "labels", "ref"]),
            ("mouse", ["acronyms", "conn", "coords", "dist", "labels", "ref"]),
            ("rat", ["conn", "labels", "ref"]),
        ],
    )
    def test_fetch_famous_gmat(self, tmp_path, dataset, expected):
        """Test fetching of famous G.mat datasets."""
        connectome = datasets.fetch_famous_gmat(dataset, data_dir=tmp_path, verbose=0)

        expected.remove("ref")
        for key in expected:
            assert key in connectome
            assert isinstance(connectome[key], str if key == "ref" else np.ndarray)
