import yaml
from pathlib import Path

channels = {}
channels['right_occipital'] = ['MEG2031', 'MEG2032', 'MEG2033', 'MEG2341', 'MEG2342', 'MEG2343', 'MEG2131', 'MEG2132',
                               'MEG2523', 'MEG2121', 'MEG2122', 'MEG2123']
channels['left_occipital'] = ['MEG2041', 'MEG2042', 'MEG2043', 'MEG1911', 'MEG1912', 'MEG1913', 'MEG1941', 'MEG1942',
                              'MEG1943', 'MEG1641', 'MEG1642', 'MEG1643', 'MEG1721', 'MEG1722', 'MEG1723', 'MEG1711',
                              'MEG1712', 'MEG1713', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1921', 'MEG1922', 'MEG1923',
                              'MEG2111', 'MEG2112', 'MEG2113', 'MEG1931', 'MEG1932', 'MEG1933', 'MEG1741', 'MEG1742',
                              'MEG1743', 'MEG2141', 'MEG2142', 'MEG2143']
channels['right_parietal'] = ['MEG0731', 'MEG0732', 'MEG0733', 'MEG2241', 'MEG2242', 'MEG2243', 'MEG2021', 'MEG2022',
                              'MEG2023', 'MEG2211', 'MEG2212', 'MEG2213', 'MEG2231', 'MEG2232', 'MEG2233', 'MEG2221',
                              'MEG2222', 'MEG2223', 'MEG2441', 'MEG2442', 'MEG2443', 'MEG0721', 'MEG0722', 'MEG0723',
                              'MEG1041', 'MEG1042', 'MEG1043', 'MEG1141', 'MEG1142', 'MEG1143', 'MEG1111', 'MEG1112',
                              'MEG1113', 'MEG1121', 'MEG1122', 'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133']
channels['left_parietal'] = ['MEG0711', 'MEG0712', 'MEG0713', 'MEG0631', 'MEG0632', 'MEG0633', 'MEG0431', 'MEG0432',
                             'MEG0433', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0411',
                             'MEG0412', 'MEG0413', 'MEG0741', 'MEG0742', 'MEG0743', 'MEG1831', 'MEG1832', 'MEG1833',
                             'MEG2011', 'MEG2012', 'MEG2013', 'MEG1821', 'MEG1822', 'MEG1823', 'MEG1841', 'MEG1842',
                             'MEG1843', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1631', 'MEG1632', 'MEG1633']
channels['right_temporal'] = ['MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 'MEG1441', 'MEG1442',
                              'MEG1443', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1341', 'MEG1342', 'MEG1343', 'MEG1331',
                              'MEG1332', 'MEG1333', 'MEG2611', 'MEG2612', 'MEG2613', 'MEG2411', 'MEG2412', 'MEG2413',
                              'MEG2421', 'MEG2422', 'MEG2423', 'MEG2641', 'MEG2642', 'MEG2643', 'MEG2631', 'MEG2632',
                              'MEG2633', 'MEG1421', 'MEG1422', 'MEG1423', 'MEG2621', 'MEG2622', 'MEG2623']
channels['left_temporal'] = ['MEG0221', 'MEG0222', 'MEG0223', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0131', 'MEG0132',
                             'MEG0133', 'MEG0231', 'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG1541',
                             'MEG1542', 'MEG1543', 'MEG1511', 'MEG1512', 'MEG1513', 'MEG1621', 'MEG1622', 'MEG1623',
                             'MEG1611', 'MEG1612', 'MEG1613', 'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532',
                             'MEG1533', 'MEG0111', 'MEG0112', 'MEG0113', 'MEG0141', 'MEG0142', 'MEG0143']
channels['right_frontal'] = ['MEG1411', 'MEG1412', 'MEG1413', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232',
                             'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1031', 'MEG1032', 'MEG1033', 'MEG1011',
                             'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG0931', 'MEG0932', 'MEG0933',
                             'MEG1211', 'MEG1212', 'MEG1213', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG0921', 'MEG0922',
                             'MEG0923', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0811', 'MEG0812', 'MEG0813']
channels['left_frontal'] = ['MEG0621', 'MEG0622', 'MEG0623', 'MEG0331', 'MEG0332', 'MEG0333', 'MEG0321', 'MEG0322',
                            'MEG0323', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0121', 'MEG0122', 'MEG0123', 'MEG0611',
                            'MEG0612', 'MEG0613', 'MEG0541', 'MEG0542', 'MEG0543', 'MEG0311', 'MEG0312', 'MEG0313',
                            'MEG0821', 'MEG0822', 'MEG0823', 'MEG0531', 'MEG0532', 'MEG0533', 'MEG0521', 'MEG0522',
                            'MEG0523', 'MEG0511', 'MEG0512', 'MEG0513', 'MEG0641', 'MEG0642', 'MEG0643']
channels['frontal'] = channels['left_frontal'] + channels['right_frontal']
channels['parietal'] = channels['left_parietal'] + channels['right_parietal']
channels['occipital'] = channels['left_occipital'] + channels['right_occipital']
channels['temporal'] = channels['left_temporal'] + channels['right_temporal']
channels['left'] = channels['left_occipital'] + channels['left_parietal'] + channels['left_frontal'] + channels[
    'left_temporal']
channels['right'] = channels['right_occipital'] + channels['right_parietal'] + channels['right_frontal'] + channels[
    'right_temporal']
channels['all'] = channels['left'] + channels['right']

channels['grad_longitude'] = ['MEG0113', 'MEG0122', 'MEG0132', 'MEG0143', 'MEG0213', 'MEG0222', 'MEG0232', 'MEG0243',
                              'MEG0313', 'MEG0322', 'MEG0333', 'MEG0343', 'MEG0413', 'MEG0422', 'MEG0432', 'MEG0443',
                              'MEG0513', 'MEG0523', 'MEG0532', 'MEG0542', 'MEG0613', 'MEG0622', 'MEG0633', 'MEG0642',
                              'MEG0713', 'MEG0723', 'MEG0733', 'MEG0743', 'MEG0813', 'MEG0822', 'MEG0913', 'MEG0923',
                              'MEG0932', 'MEG0942', 'MEG1013', 'MEG1023', 'MEG1032', 'MEG1043', 'MEG1112', 'MEG1123',
                              'MEG1133', 'MEG1142', 'MEG1213', 'MEG1223', 'MEG1232', 'MEG1243', 'MEG1312', 'MEG1323',
                              'MEG1333', 'MEG1342', 'MEG1412', 'MEG1423', 'MEG1433', 'MEG1442', 'MEG1512', 'MEG1522',
                              'MEG1533', 'MEG1543', 'MEG1613', 'MEG1622', 'MEG1632', 'MEG1643', 'MEG1713', 'MEG1722',
                              'MEG1732', 'MEG1743', 'MEG1813', 'MEG1822', 'MEG1832', 'MEG1843', 'MEG1912', 'MEG1923',
                              'MEG1932', 'MEG1943', 'MEG2013', 'MEG2023', 'MEG2032', 'MEG2042', 'MEG2113', 'MEG2122',
                              'MEG2133', 'MEG2143', 'MEG2212', 'MEG2223', 'MEG2233', 'MEG2242', 'MEG2312', 'MEG2323',
                              'MEG2332', 'MEG2343', 'MEG2412', 'MEG2423', 'MEG2433', 'MEG2442', 'MEG2512', 'MEG2522',
                              'MEG2533', 'MEG2543', 'MEG2612', 'MEG2623', 'MEG2633', 'MEG2642']

channels['grad_lattitude'] = ['MEG0112', 'MEG0123', 'MEG0133', 'MEG0142', 'MEG0212', 'MEG0223', 'MEG0233', 'MEG0242',
                              'MEG0312', 'MEG0323', 'MEG0332', 'MEG0342', 'MEG0412', 'MEG0423', 'MEG0433', 'MEG0442',
                              'MEG0512', 'MEG0522', 'MEG0533', 'MEG0543', 'MEG0612', 'MEG0623', 'MEG0632', 'MEG0643',
                              'MEG0712', 'MEG0722', 'MEG0732', 'MEG0742', 'MEG0812', 'MEG0823', 'MEG0912', 'MEG0922',
                              'MEG0933', 'MEG0943', 'MEG1012', 'MEG1022', 'MEG1033', 'MEG1042', 'MEG1113', 'MEG1122',
                              'MEG1132', 'MEG1143', 'MEG1212', 'MEG1222', 'MEG1233', 'MEG1242', 'MEG1313', 'MEG1322',
                              'MEG1332', 'MEG1343', 'MEG1413', 'MEG1422', 'MEG1432', 'MEG1443', 'MEG1513', 'MEG1523',
                              'MEG1532', 'MEG1542', 'MEG1612', 'MEG1623', 'MEG1633', 'MEG1642', 'MEG1712', 'MEG1723',
                              'MEG1733', 'MEG1742', 'MEG1812', 'MEG1823', 'MEG1833', 'MEG1842', 'MEG1913', 'MEG1922',
                              'MEG1933', 'MEG1942', 'MEG2012', 'MEG2022', 'MEG2033', 'MEG2043', 'MEG2112', 'MEG2123',
                              'MEG2132', 'MEG2142', 'MEG2213', 'MEG2222', 'MEG2232', 'MEG2243', 'MEG2313', 'MEG2322',
                              'MEG2333', 'MEG2342', 'MEG2413', 'MEG2422', 'MEG2432', 'MEG2443', 'MEG2513', 'MEG2523',
                              'MEG2532', 'MEG2542', 'MEG2613', 'MEG2622', 'MEG2632', 'MEG2643']

channels['mag'] = ['MEG0111', 'MEG0121', 'MEG0131', 'MEG0141', 'MEG0211', 'MEG0221', 'MEG0231', 'MEG0241', 'MEG0311',
                   'MEG0321', 'MEG0331', 'MEG0341', 'MEG0411', 'MEG0421', 'MEG0431', 'MEG0441', 'MEG0511', 'MEG0521',
                   'MEG0531', 'MEG0541', 'MEG0611', 'MEG0621', 'MEG0631', 'MEG0641', 'MEG0711', 'MEG0721', 'MEG0731',
                   'MEG0741', 'MEG0811', 'MEG0821', 'MEG0911', 'MEG0921', 'MEG0931', 'MEG0941', 'MEG1011', 'MEG1021',
                   'MEG1031', 'MEG1041', 'MEG1111', 'MEG1121', 'MEG1131', 'MEG1141', 'MEG1211', 'MEG1221', 'MEG1231',
                   'MEG1241', 'MEG1311', 'MEG1321', 'MEG1331', 'MEG1341', 'MEG1411', 'MEG1421', 'MEG1431', 'MEG1441',
                   'MEG1511', 'MEG1521', 'MEG1531', 'MEG1541', 'MEG1611', 'MEG1621', 'MEG1631', 'MEG1641', 'MEG1711',
                   'MEG1721', 'MEG1731', 'MEG1741', 'MEG1811', 'MEG1821', 'MEG1831', 'MEG1841', 'MEG1911', 'MEG1921',
                   'MEG1931', 'MEG1941', 'MEG2011', 'MEG2021', 'MEG2031', 'MEG2041', 'MEG2111', 'MEG2121', 'MEG2131',
                   'MEG2141', 'MEG2211', 'MEG2221', 'MEG2231', 'MEG2241', 'MEG2311', 'MEG2321', 'MEG2331', 'MEG2341',
                   'MEG2411', 'MEG2421', 'MEG2431', 'MEG2441', 'MEG2511', 'MEG2521', 'MEG2531', 'MEG2541', 'MEG2611',
                   'MEG2621', 'MEG2631', 'MEG2641']

for i in channels.keys():
    channels[i] = sorted(channels[i])

with open(str(Path(__file__).parent) + '/neuromag306_info.yml', 'w') as outfile:
    yaml.dump(channels, outfile, default_flow_style=False)

