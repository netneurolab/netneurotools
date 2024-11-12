"""For testing netneurotools.datasets.fetch_* functionality."""
import os
import pytest
from pathlib import Path
import numpy as np
from netneurotools import datasets


class TestFetchTemplate:
    """Test fetching of template datasets."""

    @pytest.mark.parametrize('version', [
        'fsaverage', 'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6'
    ])
    def test_fetch_fsaverage(self, tmpdir, version):
        """Test fetching of fsaverage surfaces."""
        fsaverage = datasets.fetch_fsaverage(
            version=version, data_dir=tmpdir, verbose=0
        )
        for k in ['orig', 'white', 'smoothwm', 'pial', 'inflated', 'sphere']:
            assert hasattr(fsaverage, k)
            assert len(fsaverage[k]) == 2
            assert all(os.path.isfile(hemi) for hemi in fsaverage[k]), fsaverage[k]

    def test_fetch_hcp_standards(self, tmpdir):
        """Test fetching of HCP standard meshes."""
        hcp = datasets.fetch_hcp_standards(data_dir=tmpdir, verbose=0)
        assert hcp.exists()

    @pytest.mark.parametrize('version', [
        'v1', 'v2'
    ])
    def test_fetch_civet(self, tmpdir, version):
        """Test fetching of CIVET templates."""
        civet = datasets.fetch_civet(version=version, data_dir=tmpdir, verbose=0)
        for key in ('mid', 'white'):
            assert key in civet
            for hemi in ('lh', 'rh'):
                assert hasattr(civet[key], hemi)
                assert os.path.isfile(getattr(civet[key], hemi))

    def test_fetch_conte69(self, tmpdir):
        """Test fetching of Conte69 surfaces."""
        conte = datasets.fetch_conte69(data_dir=tmpdir, verbose=0)
        assert all(hasattr(conte, k) for k in
                ['midthickness', 'inflated', 'vinflated', 'info'])

    def test_fetch_yerkes19(self, tmpdir):
        """Test fetching of Yerkes19 surfaces."""
        yerkes19 = datasets.fetch_yerkes19(data_dir=tmpdir, verbose=0)
        assert all(hasattr(yerkes19, k) for k in
                ['midthickness', 'inflated', 'vinflated'])


class TestFetchAtlas:
    """Test fetching of atlas datasets."""

    @pytest.mark.parametrize('version, expected', [
        ('MNI152NLin2009aSym', [1, 1, 1, 1, 1]),
        ('fsaverage', [2, 2, 2, 2, 2]),
        ('fsaverage5', [2, 2, 2, 2, 2]),
        ('fsaverage6', [2, 2, 2, 2, 2]),
        ('fslr32k', [2, 2, 2, 2, 2]),
        ('gcs', [2, 2, 2, 2, 6])
    ])
    def test_fetch_cammoun2012(self, tmpdir, version, expected):
        """Test fetching of Cammoun2012 parcellations."""
        keys = ['scale033', 'scale060', 'scale125', 'scale250', 'scale500']
        cammoun = datasets.fetch_cammoun2012(version, data_dir=tmpdir, verbose=0)

        # output has expected keys
        assert all(hasattr(cammoun, k) for k in keys)
        # and keys are expected lengths!
        for k, e in zip(keys, expected):
            out = getattr(cammoun, k)
            if isinstance(out, (tuple, list)):
                assert len(out) == e
            else:
                assert isinstance(out, Path) and str(out).endswith('.nii.gz')

    @pytest.mark.parametrize('version', [
        'fsaverage', 'fsaverage5', 'fsaverage6', 'fslr32k'
    ])
    def test_fetch_schaefer2018(self, tmpdir, version):
        """Test fetching of Schaefer2018 parcellations."""
        keys = [
            f'{p}Parcels{n}Networks'
            for p in range(100, 1001, 100) for n in [7, 17]
        ]
        schaefer = datasets.fetch_schaefer2018(version, data_dir=tmpdir, verbose=0)

        if version == 'fslr32k':
            assert all(k in schaefer and os.path.isfile(schaefer[k]) for k in keys)
        else:
            for k in keys:
                assert k in schaefer
                assert len(schaefer[k]) == 2
                assert all(os.path.isfile(hemi) for hemi in schaefer[k])

    def test_fetch_mmpall(self, tmpdir):
        """Test fetching of MMPAll parcellations."""
        mmp = datasets.fetch_mmpall(data_dir=tmpdir, verbose=0)
        assert len(mmp) == 2
        assert all(os.path.isfile(hemi) for hemi in mmp)
        assert all(hasattr(mmp, attr) for attr in ('lh', 'rh'))

    def test_fetch_pauli2018(self, tmpdir):
        """Test fetching of Pauli2018 parcellations."""
        pauli = datasets.fetch_pauli2018(data_dir=tmpdir, verbose=0)
        assert all(hasattr(pauli, k) and os.path.isfile(pauli[k]) for k in
                ['probabilistic', 'deterministic', 'info'])

    @pytest.mark.xfail
    def test_fetch_ye2020(self, tmpdir):
        """Test fetching of Ye2020 parcellations."""
        assert False

    def test_fetch_voneconomo(self, tmpdir):
        """Test fetching of von Economo parcellations."""
        vek = datasets.fetch_voneconomo(data_dir=tmpdir, verbose=0)
        assert all(hasattr(vek, k) and len(vek[k]) == 2 for k in ['gcs', 'ctab'])
        assert isinstance(vek.get('info'), Path)


class TestFetchProject:
    """Test fetching of project datasets."""

    def test_fetch_vazquez_rodriguez2019(self, tmpdir):
        """Test fetching of Vazquez-Rodriguez2019 dataset."""
        vazquez = datasets.fetch_vazquez_rodriguez2019(data_dir=tmpdir, verbose=0)
        for k in ['rsquared', 'gradient']:
            assert hasattr(vazquez, k)
            assert isinstance(getattr(vazquez, k), np.ndarray)

    @pytest.mark.xfail
    def test_fetch_mirchi2018(self, tmpdir):
        """Test fetching of Mirchi2018 dataset."""
        assert False

    def test_fetch_hansen_manynetworks(self, tmpdir):
        """Test fetching of Hansen et al., 2023 many-networks dataset."""
        hansen = datasets.fetch_hansen_manynetworks(data_dir=tmpdir, verbose=0)
        assert "cammoun033" in hansen
        assert "gene" in hansen["cammoun033"]
        assert isinstance(hansen["cammoun033"]["gene"], Path)

    @pytest.mark.xfail
    def test_fetch_hansen_receptors(self, tmpdir):
        """Test fetching of Hansen et al., 2022 receptor dataset."""
        # hansen = datasets.fetch_hansen_receptors(data_dir=tmpdir, verbose=0)
        assert False

    @pytest.mark.xfail
    def test_fetch_hansen_genecognition(self, tmpdir):
        """Test fetching of Hansen et al., 2021 gene-cognition dataset."""
        # hansen = datasets.fetch_hansen_genecognition(data_dir=tmpdir, verbose=0)
        assert False

    @pytest.mark.xfail
    def test_fetch_hansen_brainstem(self, tmpdir):
        """Test fetching of Hansen et al., 2024 brainstem dataset."""
        # hansen = datasets.fetch_hansen_brainstem(data_dir=tmpdir, verbose=0)
        assert False

    @pytest.mark.xfail
    def test_fetch_shafiei_megfmrimapping(self, tmpdir):
        """Test fetching of Shafiei et al., 2022 & 2023 HCP-MEG dataset."""
        # shafiei = datasets.fetch_shafiei_megfmrimapping(data_dir=tmpdir, verbose=0)
        assert False

    @pytest.mark.xfail
    def test_fetch_shafiei_megdynamics(self, tmpdir):
        """Test fetching of Shafiei et al., 2022 & 2023 HCP-MEG dataset."""
        # shafiei = datasets.fetch_shafiei_megdynamics(data_dir=tmpdir, verbose=0)
        assert False

    @pytest.mark.xfail
    def test_fetch_suarez_mami(self, tmpdir):
        """Test fetching of Suarez et al., 2022 mami dataset."""
        # suarez = datasets.fetch_suarez_mami(data_dir=tmpdir, verbose=0)
        assert False

    @pytest.mark.parametrize('dataset, expected', [
        ('celegans', ['conn', 'dist', 'labels', 'ref']),
        ('drosophila', ['conn', 'coords', 'labels', 'networks', 'ref']),
        ('human_func_scale033', ['conn', 'coords', 'labels', 'ref']),
        ('human_func_scale060', ['conn', 'coords', 'labels', 'ref']),
        ('human_func_scale125', ['conn', 'coords', 'labels', 'ref']),
        ('human_func_scale250', ['conn', 'coords', 'labels', 'ref']),
        ('human_func_scale500', ['conn', 'coords', 'labels', 'ref']),
        ('human_struct_scale033', ['conn', 'coords', 'dist', 'labels', 'ref']),
        ('human_struct_scale060', ['conn', 'coords', 'dist', 'labels', 'ref']),
        ('human_struct_scale125', ['conn', 'coords', 'dist', 'labels', 'ref']),
        ('human_struct_scale250', ['conn', 'coords', 'dist', 'labels', 'ref']),
        ('human_struct_scale500', ['conn', 'coords', 'dist', 'labels', 'ref']),
        ('macaque_markov', ['conn', 'dist', 'labels', 'ref']),
        ('macaque_modha', ['conn', 'coords', 'dist', 'labels', 'ref']),
        ('mouse', ['acronyms', 'conn', 'coords', 'dist', 'labels', 'ref']),
        ('rat', ['conn', 'labels', 'ref']),
    ])
    def test_fetch_famous_gmat(self, tmpdir, dataset, expected):
        """Test fetching of famous G.mat datasets."""
        connectome = datasets.fetch_famous_gmat(dataset, data_dir=tmpdir, verbose=0)

        expected.remove("ref")
        for key in expected:
            assert (key in connectome)
            assert isinstance(connectome[key], str if key == 'ref' else np.ndarray)
