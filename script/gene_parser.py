import gzip
import os
import pandas as pd
import argparse

# to run: ensure you are running Python 3.6.x and Pandas 0.20.x
# python gene_parser.py --file=file_name.gz
# parser will create .csv file in exome/processed subfolder
# rows = subjects (GRID), columns = gene


def print_list_head(lst, num):
    if num > len(lst):
        raise Exception('num ({}) is greater than list size ({})'.format(num, len(lst)))

    str = ''
    for i in range(num):
        str += '{}\t'.format(lst[i])

    print(str)


parser = argparse.ArgumentParser()
parser.add_argument('--file', action='store', dest='file', type=str)

cmd_arg = parser.parse_args()

exome_path = '/legodata/predixcan/CEU_23K'

file = cmd_arg.file

with gzip.open(os.path.join(exome_path, file), 'rt') as f:
    file_content = f.read()

file_content_lst = file_content.split('\n')

subj_lst = file_content_lst.pop(0).split('\t')

exome_data = list()
for str in file_content_lst:
    gene_data = str.split('\t')

    if len(gene_data) > 1:
        exome_data.append(gene_data)

df_exome_data = pd.DataFrame(exome_data)
df_exome_data.columns = subj_lst

df_exome_data.pop('##predict_exists_rs_num|exists_rs_num')
df_exome_data = df_exome_data.set_index('#PID')
df_exome_data = df_exome_data.transpose()

df_exome_data = df_exome_data.reset_index()
df_exome_data = df_exome_data.rename(columns={'index': 'grid'})

processed_path = '/legodata/exome/processed'
df_exome_data.to_csv(os.path.join(processed_path, file + '.csv'), index=False)
