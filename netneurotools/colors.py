# -*- coding: utf-8 -*-
"""Useful colormaps."""

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

__all__ = ['parula', 'justine', 'dinosaur']


parula = LinearSegmentedColormap.from_list(
    'parula',
    ['#352A87', '#363093', '#3637A0', '#353DAD', '#3243BA', '#2C4AC7',
     '#2053D4', '#0F5CDD', '#0363E1', '#0268E1', '#046DE0', '#0871DE',
     '#0D75DC', '#1079DA', '#127DD8', '#1481D6', '#1485D4', '#1389D3',
     '#108ED2', '#0C93D2', '#0998D1', '#079CCF', '#06A0CD', '#06A4CA',
     '#06A7C6', '#07A9C2', '#0AACBE', '#0FAEB9', '#15B1B4', '#1DB3AF',
     '#25B5A9', '#2EB7A4', '#38B99E', '#42BB98', '#4DBC92', '#59BD8C',
     '#65BE86', '#71BF80', '#7CBF7B', '#87BF77', '#92BF73', '#9CBF6F',
     '#A5BE6B', '#AEBE67', '#B7BD64', '#C0BC60', '#C8BC5D', '#D1BB59',
     '#D9BA56', '#E1B952', '#E9B94E', '#F1B94A', '#F8BB44', '#FDBE3D',
     '#FFC337', '#FEC832', '#FCCE2E', '#FAD32A', '#F7D826', '#F5DE21',
     '#F5E41D', '#F5EB18', '#F6F313', '#F9FB0E']
)


justine = ListedColormap(
    ['#3855A5', '#3857A6', '#3958A8', '#395AA9', '#395BAB', '#3A5DAC',
     '#3A5EAD', '#3A60AF', '#3B61B0', '#3B63B1', '#3B65B3', '#3C66B4',
     '#3C68B6', '#3D69B7', '#3D6BB8', '#3D6CBA', '#3E6EBB', '#3E6FBC',
     '#3E71BE', '#3F73BF', '#3F74C1', '#3F76C2', '#4077C3', '#4079C5',
     '#407AC6', '#417CC8', '#417DC9', '#417FCA', '#4281CC', '#4282CD',
     '#4284CE', '#4385D0', '#4387D1', '#4488D3', '#448AD4', '#448BD5',
     '#458DD7', '#458FD8', '#4590D9', '#4692DB', '#4693DC', '#4695DE',
     '#4796DF', '#4798E0', '#4799E2', '#489BE3', '#489DE5', '#489EE6',
     '#49A0E7', '#49A1E9', '#49A3EA', '#4AA4EB', '#4AA6ED', '#4BA7EE',
     '#4BA9F0', '#4BABF1', '#4CACF2', '#4CAEF4', '#4CAFF5', '#4DB1F6',
     '#4DB2F8', '#4DB4F9', '#4EB5FB', '#4EB7FC', '#51B8FC', '#54B9FC',
     '#56BAFC', '#59BBFC', '#5CBDFC', '#5FBEFC', '#61BFFC', '#64C0FC',
     '#67C1FC', '#6AC2FC', '#6CC3FD', '#6FC5FD', '#72C6FD', '#75C7FD',
     '#77C8FD', '#7AC9FD', '#7DCAFD', '#80CBFD', '#83CCFD', '#85CDFD',
     '#88CFFD', '#8BD0FD', '#8ED1FD', '#90D2FD', '#93D3FD', '#96D4FD',
     '#99D5FD', '#9BD7FD', '#9ED8FD', '#A1D9FD', '#A4DAFD', '#A6DBFE',
     '#A9DCFE', '#ACDDFE', '#AFDEFE', '#B2DFFE', '#B4E1FE', '#B7E2FE',
     '#BAE3FE', '#BDE4FE', '#BFE5FE', '#C2E6FE', '#C5E7FE', '#C8E8FE',
     '#CAEAFE', '#CDEBFE', '#D0ECFE', '#D3EDFE', '#D6EEFE', '#D8EFFE',
     '#DBF0FE', '#DEF2FE', '#E1F3FE', '#E3F4FF', '#E6F5FF', '#E9F6FF',
     '#ECF7FF', '#EEF8FF', '#F1F9FF', '#F4FAFF', '#F7FCFF', '#F9FDFF',
     '#FCFEFF', '#FFFFFF', '#FFFEFC', '#FFFDF9', '#FFFCF7', '#FFFBF4',
     '#FFF9F1', '#FFF8EE', '#FFF7EC', '#FFF6E9', '#FFF5E6', '#FFF4E3',
     '#FFF3E1', '#FFF2DE', '#FFF0DB', '#FFEFD8', '#FFEED6', '#FFEDD3',
     '#FFECD0', '#FFEBCD', '#FFEACB', '#FFE9C8', '#FFE7C5', '#FFE6C2',
     '#FFE5C0', '#FFE4BD', '#FFE3BA', '#FFE2B7', '#FFE1B5', '#FFE0B2',
     '#FFDFAF', '#FFDDAC', '#FFDCAA', '#FFDBA7', '#FFD9A3', '#FFD79F',
     '#FFD59C', '#FFD398', '#FFD194', '#FECF90', '#FECE8C', '#FECC88',
     '#FECA85', '#FEC881', '#FEC67D', '#FEC479', '#FEC275', '#FEC072',
     '#FEBE6E', '#FEBC6A', '#FEBA66', '#FDB862', '#FDB65F', '#FDB45B',
     '#FDB257', '#FDB053', '#FDAE4F', '#FDAC4C', '#FDAA48', '#FDA844',
     '#FDA740', '#FDA53C', '#FDA339', '#FCA135', '#FC9F31', '#FC9D2D',
     '#FC9B29', '#FC9926', '#FC9722', '#FC951E', '#FB921E', '#FB901F',
     '#FA8D1F', '#F98A1F', '#F9871F', '#F88520', '#F88220', '#F77F20',
     '#F67D21', '#F67A21', '#F57721', '#F47521', '#F47222', '#F36F22',
     '#F26C22', '#F26A22', '#F16723', '#F16423', '#F06223', '#EF5F24',
     '#EF5C24', '#EE5A24', '#ED5724', '#ED5425', '#EC5125', '#EB4F25',
     '#EB4C26', '#EA4926', '#EA4726', '#E94426', '#E84127', '#E83E27',
     '#E73C27', '#E63928', '#E63628', '#E53428', '#E43128', '#E42E29',
     '#E32C29', '#E22929', '#E22629', '#DF2528', '#DD2427', '#DB2326',
     '#D82225', '#D62123', '#D42022', '#D11F21', '#CF1E20', '#CD1D1E',
     '#CA1C1D', '#C81A1C', '#C6191B', '#C31819', '#C11718', '#BE1617',
     '#BC1516', '#BA1414', '#B71313', '#B51212'],
    'justine'
)


# https://doi.org/10.1038/s41586-022-04770-6
dinosaur = LinearSegmentedColormap.from_list(
    'dinosaur',
    ['#02B2CE', '#0DB3C4', '#18B4BB', '#24B6B2', '#2FB7A9', '#3AB8A0',
     '#46BA97', '#51BB8E', '#5CBC85', '#68BE7C', '#73BF73', '#7EC06A',
     '#8AC261', '#95C358', '#A1C44F', '#ACC645', '#B7C73C', '#C3C833',
     '#CECA2A', '#D9CB21', '#E5CC18', '#F0CE0F', '#FBCF06', '#FECD04',
     '#FDC805', '#FDC405', '#FCC006', '#FBBC07', '#FBB807', '#FAB408',
     '#FAB009', '#F9AC09', '#F8A80A', '#F8A40B', '#F79F0C', '#F69B0C',
     '#F6970D', '#F5930E', '#F48F0E', '#F48B0F', '#F38710', '#F38310',
     '#F27F11', '#F17B12', '#F17612', '#F07213', '#EF6E14', '#EF6A14',
     '#EE6615', '#ED6216', '#ED5E17', '#EC5A17', '#EC5618', '#EB5219',
     '#EA4D19', '#EA491A', '#E9451B', '#E8411B', '#E83D1C', '#E7391D',
     '#E6351D', '#E6311E', '#E52D1F', '#E52920']
)


def available_cmaps():
    """Return list of available colormaps in module."""
    return __all__.copy()


def _register_cmaps():
    """Register all colormaps in module so they are accessible via matplotlib."""
    for cmap in __all__:
        matplotlib.colormaps.register(globals()[cmap], name=cmap)


_register_cmaps()
