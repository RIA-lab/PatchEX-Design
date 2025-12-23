import pandas as pd
import numpy as np
from tqdm import tqdm

gb1_df = pd.read_excel('GB1.xlsx')
gb1_data = gb1_df[['Variants', 'Fitness']].values.tolist()

max_fit = np.max([d[1] for d in gb1_data])

with open('gb1.csv', 'w', encoding='utf-8') as file:
    file.write('seq,score\n')
    for data in tqdm(gb1_data):
        file.write(f'{data[0]},{data[1]/max_fit}\n')


phoq_df = pd.read_excel('PhoQ.xlsx')
phoq_data = phoq_df[['Variants', 'Fitness']].values.tolist()
max_fit = np.max([d[1] for d in phoq_data])

with open('phoq.csv', 'w', encoding='utf-8') as file:
    file.write('seq,score\n')
    for data in tqdm(phoq_data):
        file.write(f'{data[0]},{data[1]/max_fit}\n')