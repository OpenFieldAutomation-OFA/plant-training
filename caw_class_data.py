import argparse
import tqdm
import os
# from utilities import datasets
# from map_dataset import map_dataset

# The expected folder layout of the caw_data directory is:
# - caw_data
#   - images
#   - bboxes
#   - params
#   - LICENCE.txt
#   - README.txt

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',
                    default="/mnt/c/caw/download",
                    help="Directory of the downloaded CAW data.")
parser.add_argument('--output_dir',
                    default="/mnt/c/caw/classification",
                    help="Path to the directory to save classification data.")

args = parser.parse_args()

# some CAW classes only specify the genus
# some CAW classes are not present in plantclef
labelid_to_speciesid = {
    # 0: Soil
    1: 1363500, # Zea mays
    2: 1363500, # Zea mays
    3: 1363500, # Zea mays
    4: 1363500, # Zea mays
    5: 1363500, # Zea mays
    6: 1363500, # Zea mays
    7: 1363199, # Beta vulgaris
    8: 1363199, # Beta vulgaris
    9: 1363199, # Beta vulgaris
    10: 1363199, # Beta vulgaris
    11: 1363199, # Beta vulgaris
    12: 1363199, # Beta vulgaris
    13: 1741625, # Lathyrus oleraceus
    14: 1384485, # Cucurbita pepo
    15: 1384485, # Cucurbita pepo (CAW only genus)
    # 16: Radish
    17: 1363268, # Asparagus officinalis
    18: 1356171, # Solanum tuberosum
    19: 1363214, # Petroselinum crispum
    20: 1363214, # Petroselinum crispum
    21: 1360145, # Primula veris
    # 22: Poppy
    23: 1356780, # Cannabis sativa
    24: 1364145, # Helianthus annuus
    # 25: Sage
    # 26: Common Bean
    27: 1363676, # Vicia faba
    28: 1359517, # Trifolium pratense
    29: 1389975, # Chenopodiastrum hybridum
    30: 1392250, # Fallopia convolvulus
    31: 1391823, # Echinochloa crus-galli
    32: 1356476, # Amaranthus retroflexus
    33: 1363173, # Chenopodium album
    34: 1360549, # Datura stramonium
    35: 1355935, # Galinsoga parviflora
    36: 1362332, # Matricaria chamomilla
    # 37: Saltbush
    38: 1355936, # Cirsium arvense
    39: 1362380, # Sonchus arvensis
    40: 1360141, # Portulaca oleracea
    41: 1356167, # Solanum nigrum
    42: 1358605, # Mercurialis perennis  (CAW only genus)
    43: 1392155, # Euphorbia cyparissias (CAW only genus)
    44: 1356091, # Persicaria lapathifolia
    45: 1356017, # Geranium robertianum (CAW only genus)
    46: 1360385, # Galium aparine
    47: 1358347, # Lepidium draba
    48: 1363458, # Poa pratensis (CAW only genus)
    49: 1363181, # Atriplex laciniata
    50: 1358722, # Ballota nigra
    51: 1363910, # Capsella bursa-pastoris
    52: 1364006, # Convolvulus arvensis
    53: 1364152, # Artemisia vulgaris
    54: 1363897, # Sisymbrium officinale
    55: 1364138, # Senecio vulgaris
    56: 1397521, # Veronica chamaedrys (CAW only genus)
    57: 1356067, # Plantago major
    58: 1358204, # Calepina irregularis
    59: 1560691, # Mentha × piperita
    60: 1363886, # Thlaspi arvense
    61: 1356538, # Spergula arvensis
    62: 1363424, # Digitaria sanguinalis
    63: 1359792, # Fumaria officinalis
    64: 1364064, # Veronica hederifolia
    65: 1363455, # Poa annua
    66: 1356094, # Persicaria maculosa
    67: 1392433, # Galeopsis tetrahit
    68: 1363459, # Poa trivialis
    69: 1425584, # Setaria viridis
    70: 1392594, # Geranium pusillum
    71: 1411387, # Centaurea cyanus
    72: 1356672, # Agrostemma githago
    73: 1360222, # Ranunculus repens
    74: 1361307, # Hordeum murinum
    75: 1720968, # Festuca myuros
    76: 1358752, # Lamium purpureum
    77: 1356066, # Plantago lanceolata
    78: 1357711, # Matricaria discoidea
    79: 1356571, # Stellaria media
    80: 1363897, # Sisymbrium officinale
    81: 1356286, # Bromus hordeaceus
    82: 1356215, # Viola tricolor
    83: 1728571, # Barbarea vulgaris
    84: 1363468, # Avena fatua
    85: 1363128, # Papaver rhoeas
    86: 1361206, # Bromus secalinus
    87: 1356084, # Polygonum aviculare
    88: 1364173, # Lactuca serriola
    89: 1392251, # Fallopia dumetorum
    90: 1389976, # Lipandra polysperma
    91: 1356095, # Fagopyrum esculentum
    92: 1392774, # Allium schoenoprasum 
    93: 1358126, # Allium sativum
    # 94: Soybean
    95: 1363227, # Daucus carota
    96: 1737463, # Rhamphospermum arvense
    97: 1363236, # Ferula communis
    98: 1356394, # Equisetum arvense
    99: 1737545, # Taraxacum sect. Taraxacum
    # 255: Vegetation
}





# if with_mapping:
#     for dataset in datasets.DATASETS:
#         if dataset != 'CropAndWeed':
#             map_dataset(os.path.join(data_root, 'bboxes'), os.path.join(data_root, 'labelIds'), 'CropAndWeed',
#                         dataset)

# for image_name in os.listdir('../images'):
#     shutil.copy(os.path.join('../images', image_name), os.path.join(data_root, 'images', image_name))


