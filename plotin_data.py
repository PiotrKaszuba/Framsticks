import pandas as pd
from datetime import datetime
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

date_format = '%Y%m%d%H%M%S'

def select_lines(file_name):
    col_names = ['num_cr', 'num_gen', 'worst', 'avg', 'best']
    df = pd.DataFrame(columns=['start', 'end', 'time', 'param', 'run'] + col_names)
    with open(file_name, 'r') as f:
        since_start = -9999
        since_finish = -9999
        obj = {}
        obj_list = []
        logged_values = []
        parameterization = 0
        run = 0
        for line in f.readlines():

            if line.startswith('C:\\'):
                regex = '[a-z]*_[0-9]+_[0-9]+'
                match = re.search(regex, line)
                if match is not None:
                    _, parameterization, run = match.group(0).split('_')

            if 'start' in line.split(' ')[0]:
                obj = {} #initialize a fresh object
                since_start = 0
            elif 'finish' in line.split(' ')[0]:
                since_finish = 0

            elif since_start is not None and since_start==3:
                date = ''.join([c for c in line if c != ' ' and c!= chr(0)]).split('.')[0]
                obj['start_time'] = datetime.strptime(date, date_format)
            elif since_finish is not None and since_finish==3:
                date = ''.join([c for c in line if c != ' ' and c!= chr(0)]).split('.')[0]
                obj['end_time'] = datetime.strptime(date, date_format)

                data = np.stack(logged_values)
                logged_values = []
                obj['time'] = obj['end_time'] - obj['start_time']

                info = [obj['start_time'], obj['end_time'], obj['time'], parameterization, run]
                data = np.apply_along_axis(lambda x: np.insert(x, 0, info), 1, data)
                new_part_df = pd.DataFrame(data, columns=df.columns)
                df = df.append(new_part_df, ignore_index=True)

                obj_list.append(obj) #save the object

            elif '[LOG]' in line:
               logged_values.append(line.split()[2:])

            since_start += 1
            since_finish += 1

        return df

def fix_dtypes(df):
    numerics = ['param', 'run', 'num_cr', 'num_gen', 'worst', 'avg', 'best']
    dates = ['start', 'end']

    for col in numerics:
        df[col] = pd.to_numeric(df[col])
    for col in dates:
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
    df['time'] = pd.to_timedelta(df['time'])
    return df

def plot_avg_fitness(df):
    sns.set(style='darkgrid')
    plot = sns.violinplot(x=df['param'], y=df['avg'])
    plot.set_title("Average scores for all parameterizations")
    plot.set_ylabel("Average score")
    plot.set_xlabel("Parameterizations")
    plt.show()

def plot_best_fitness(df):
    sns.set(style='darkgrid')
    plot = sns.violinplot(x=df['param'], y=df['best'])
    plot.set_title('Best scores for all parametrizations')
    plot.set_ylabel("Best score")
    plot.set_xlabel("Parameterizations")
    plt.show()

def plot_computation_time_for_parameterizations(df):
    sns.set(style='darkgrid')
    plot = sns.violinplot(x=df['param'], y=df['time'] / np.timedelta64(1, 's'))
    plot.set_title('Duration of experiments for all parametrizations')
    plot.set_ylabel("Duration of experiment")
    plot.set_xlabel("Parameterizations")
    plt.show()

def plot_multiline_fitness(df):
    sns.set(style='darkgrid')
    tmp_df = df[df['param']==1]
    for i, c in zip(df.param.unique(), ['Reds_r', 'Blues_r', 'Greens_r']):
        tmp_df = df[df['param']==i]

        palette = dict(zip(tmp_df.run.unique(),
                           sns.color_palette(c, 15)[:10]))
        plot = sns.relplot(x='num_cr', y='best',
                   hue='run', palette=palette, kind='line', data=tmp_df)
        plot.set_axis_labels("Number of creatures", "Best score")
        plt.title("Number of creatures and their score for parameterization {}".format(i))

    plt.show()

df = select_lines('outputFile.txt')
df = fix_dtypes(df)
plot_multiline_fitness(df)
