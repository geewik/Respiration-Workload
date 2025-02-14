from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from ast import literal_eval

from multiprocessing import Pool, cpu_count
from functools import partial

plt.close('all')

WALK_DIR = "D:/Raymond Chia/OneDrive - UTS/Data/aria-walking/"
SEAT_DIR = "D:\aria-seated"

CONDITIONS = ['R', 'L0', 'L1', 'L2', 'L3']

# For seated data load each subject and condition as a dict
def get_seated_sbj_data(sbj='S12'):
    sbj_dir = join(SEAT_DIR, sbj)
    seated_conds = ['M'] + CONDITIONS

    sens_axes = ['x', 'y', 'z']
    imu_dict, pss_dict, smm_dict = {}, {}, {}
    for cond in seated_conds:
        try:
            imu_fname = join(sbj_dir, f'{cond}_imu_df.csv')
            pss_fname = join(sbj_dir, f'{cond}_pressure_df.csv')
            smm_fname = join(sbj_dir, f'{cond}_summary_df.csv')

            imu_df = pd.read_csv(imu_fname)
            acc_data = np.vstack(
                imu_df['accelerometer'].map(literal_eval).tolist()
            )
            gyr_data = np.vstack(
                imu_df['gyroscope'].map(literal_eval).tolist()
            )
            for i, sens_axis in enumerate(sens_axes):
                acc_axis = 'acc_' + sens_axis
                gyr_axis = 'gyr_' + sens_axis

                imu_df[acc_axis] = acc_data[:, i]
                imu_df[gyr_axis] = gyr_data[:, i]

            imu_df = imu_df.drop(
                columns=['type', 'accelerometer', 'gyroscope', 'magnetometer']
            )
            imu_dict.update({cond: imu_df})

            pss_dict.update({cond: pd.read_csv(pss_fname)})
            smm_dict.update({cond: pd.read_csv(smm_fname)})
        except FileNotFoundError:
            pass

    return imu_dict, pss_dict, smm_dict

def get_walk_sbj_sensor(sbj='S12', sensor='summary'):
    sbj_dir = join(WALK_DIR, sbj, 'pkl')
    walk_conds = ['SR', 'WR'] + CONDITIONS[1:]
    sens_dict = {}

    for cond in walk_conds:
        try:
            fname = glob(join(sbj_dir, f'{cond}_*{sensor}*_{sbj}.pkl'))[0]
            df = pd.read_pickle(fname)
            sens_dict.update({cond: df})
        except FileNotFoundError:
            pass

    return sens_dict

if __name__ == '__main__':
    # load seated data
    seat_sbjs = ['S'+str(i).zfill(2) for i in range(12, 31)]
    walk_sbjs = ['S'+str(i).zfill(2) for i in range(1, 31)]

    seat_sbj = seat_sbjs[0]
    walk_sbj = walk_sbjs[14]
    
    imu_dict, pss_dict, smm_dict = get_seated_sbj_data(seat_sbj)
    bvp_dict = get_walk_sbj_sensor(sbj=walk_sbj, sensor='bvp')

    # NOTE: Uncomment me when ready to preprocess all subjects
    # with Pool(cpu_count()//2) as p:
    #     seated_data = p.map(get_seated_sbj_data, seated_sbjs)

    ''' Plotting '''
    fig, axs = plt.subplots(2, 2)

    cond0 = 'L0'
    axs[0, 0].plot(imu_dict[cond0]['sec'], imu_dict[cond0]['acc_y'],
                   label='acc$_{y}$')
    twin = plt.twinx(axs[0, 0])
    twin.plot(smm_dict[cond0]['sec'], smm_dict[cond0]['BR'],
              label='Breathing', color='tab:orange')
    axs[1, 0].plot(pss_dict[cond0]['sec'],
                   pss_dict[cond0]['Breathing Waveform'],
                  label='pressure')
    twin = plt.twinx(axs[1, 0])
    twin.plot(smm_dict[cond0]['sec'], smm_dict[cond0]['BR'],
              label='Breathing', color='tab:orange')

    cond1 = 'L3'
    axs[0, 1].plot(imu_dict[cond1]['sec'], imu_dict[cond1]['acc_y'],
                   label='acc$_{y}$')
    twin = plt.twinx(axs[0, 1])
    twin.plot(smm_dict[cond1]['sec'], smm_dict[cond1]['BR'],
              label='Breathing', color='tab:orange')
    axs[1, 1].plot(pss_dict[cond1]['sec'],
                   pss_dict[cond1]['Breathing Waveform'],
                   label='pressure')
    twin = plt.twinx(axs[1, 1])
    twin.plot(smm_dict[cond1]['sec'], smm_dict[cond1]['BR'],
              label='Breathing', color='tab:orange')

    fig, axs = plt.subplots(3, 2)
    axs = axs.flatten()
    for i, (key, val) in enumerate(bvp_dict.items()):
        axs[i].plot(val['bvp'])
        axs[i].set_title(key)

    plt.show()
