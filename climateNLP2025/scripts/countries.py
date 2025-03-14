from pathlib import Path
import pandas as pd
import difflib
import os

folder_path = Path('X:\\User\\troosts\\projects\\appl-docchat\\docs\\GBF_T7')
gbf_country_names = [f.stem for f in folder_path.glob('*.txt')]

data_path = Path('X:\\User\\troosts\\projects\\appl-docchat\\climateNLP2025\\data')
data_file = os.path.join(data_path, 'land-area-km.csv')
df_world_data = pd.read_csv(data_file, header=0)
df_world_data = df_world_data.loc[df_world_data['Year'] == 2022]
df_world_data.to_csv(os.path.join(data_path, 'world_countries.csv'), index=False)
df_world_data_country_names = df_world_data['Entity'].tolist()

# Check for each country in GBF folder which country in the data_country_names list is most similar
most_similar = {name: difflib.get_close_matches(name, df_world_data_country_names, n=1, cutoff=0) for name in gbf_country_names}
most_similar = {k: v[0] if v else None for k, v in most_similar.items()}

# write results to a csv file
df = pd.DataFrame(most_similar.items(), columns=['GBF_country', 'Data_country'])
# the result is written once, then some manual udates were appllied for matches that were wrong.
# No matches for Israel and Cook Islands!
# df.to_csv(os.path.join(data_path, 'country_mapping.csv'), index=False)

# add the mapping to the data
df_mapping = pd.read_csv(os.path.join(data_path, 'country_mapping.csv'), header=0)
df_merged = pd.merge(left=df_world_data, right=df_mapping, how='inner', left_on='Entity', right_on='Data_country')
df_merged = df_merged.drop(columns=['Data_country'])
df_merged.to_csv(os.path.join(data_path, 'world_countries_mapped.csv'), index=False)
