import pandas as pd

save_dir = './results/'
m1_name = 'dummy_results1'
m2_name = 'dummy_results2'
m3_name = 'dummy_results3'
m1 = pd.read_csv(save_dir + m1_name + '.csv')
m2 = pd.read_csv(save_dir + m2_name + '.csv')
m3 = pd.read_csv(save_dir + m3_name + '.csv')

ensem_m = m1.copy()
ensem_m['pred'] = (m1['pred'] + m2['pred'] + m3['pred']) / 3
ensem_m.to_csv(save_dir + 'results.csv', header=True, index=False)
