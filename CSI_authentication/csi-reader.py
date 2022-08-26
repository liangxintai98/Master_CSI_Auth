import numpy as np
import struct
import math
from dataclasses import dataclass
import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import OneClassSVM
from CSIKit.util.filters import running_mean


# Legacy mode
scidx_legacy = range(-26, 27)
scidx_legacy_dc = [0]
scidx_legacy_pilot = [-21, -7, 7, 21]
scidx_legacy_csi = [x for x in scidx_legacy if x not in scidx_legacy_dc]
scidx_legacy_csi_no_pilot = [x for x in scidx_legacy_csi if x not in scidx_legacy_pilot]
gen_scidx_legacy_csi_no_pilot = [scidx_legacy_csi.index(x) for x in scidx_legacy_csi_no_pilot]
sc_count_legacy = len(scidx_legacy_csi)

# Specific for AX200, pilot reported before data subcarriers.
ax200_scidx_legacy_csi = scidx_legacy_pilot + scidx_legacy_csi_no_pilot
ax200_gen_scidx_legacy_csi = [ax200_scidx_legacy_csi.index(x) for x in scidx_legacy_csi]

# HT20
scidx_20mhz = range(-28, 29)
scidx_20mhz_dc = [0]
scidx_20mhz_pilot = [-21, -7, 7, 21]
scidx_20mhz_csi = [x for x in scidx_20mhz if x not in scidx_20mhz_dc]
scidx_20mhz_csi_no_pilot = [x for x in scidx_20mhz_csi if x not in scidx_20mhz_pilot]
gen_scidx_20mhz_csi_no_pilot = [scidx_20mhz_csi.index(x) for x in scidx_20mhz_csi_no_pilot]
sc_count_20mhz = len(scidx_20mhz_csi)

# HT40
scidx_40mhz = range(-58, 59)
scidx_40mhz_dc = [-1, 0, 1]
scidx_40mhz_pilot = [-53, -25, -11, 11, 25, 53]
scidx_40mhz_csi = [x for x in scidx_40mhz if x not in scidx_40mhz_dc]
scidx_40mhz_csi_no_pilot = [x for x in scidx_40mhz_csi if x not in scidx_40mhz_pilot]
gen_scidx_40mhz_csi_no_pilot = [scidx_40mhz_csi.index(x) for x in scidx_40mhz_csi_no_pilot]
sc_count_40mhz = len(scidx_40mhz_csi)
phase_shift_scidx_40mhz = 2
phase_shift_gen_scidx_40mhz = scidx_40mhz_csi.index(phase_shift_scidx_40mhz)
phase_shift_40mhz = np.pi / 2

# 80MHz
scidx_80mhz = range(-122, 123)
scidx_80mhz_dc = [-1, 0, 1]
scidx_80mhz_pilot = [-103, -75, -39, -11, 11, 39, 75, 103]
scidx_80mhz_csi = [x for x in scidx_80mhz if x not in scidx_80mhz_dc]
scidx_80mhz_csi_no_pilot = [x for x in scidx_80mhz_csi if x not in scidx_80mhz_pilot]
gen_scidx_80mhz_csi_no_pilot = [scidx_80mhz_csi.index(x) for x in scidx_80mhz_csi_no_pilot]
sc_count_80mhz = len(scidx_80mhz_csi)
phase_shift_scidx_80mhz = -64
# phase_shift_gen_scidx_80mhz = scidx_80mhz_csi_no_pilot.index(phase_shift_scidx_80mhz)  # AC86U
phase_shift_gen_scidx_80mhz = scidx_80mhz_csi.index(phase_shift_scidx_80mhz)  # AX200
phase_shift_80mhz = np.pi

calib_step = 10 * 10 ** -9  # ns
freq_delta = 312.5 * 10 ** 3
max_slope = 5000 * 10 ** -9
min_slope = -max_slope

size_of_int = 4
num_antennas = 2
pattern = bytes([0xaa, 0xaa])

macs_filter = [
    "e0:6d:17:3f:66:b8"
]
macs_filter = [mac.lower() for mac in macs_filter]

@dataclass
class CSI:
    hdr: bytes = None
    mac: str = None
    nrx: int = None
    ntx: int = None
    num_tones: int = None
    freq: int = None
    ftmClock: int = None
    muClock: int = None
    timestamp: int = None

    data: np.ndarray = None
    phases: np.ndarray = None
    amplitude: np.ndarray = None
    calib_phases: np.ndarray = None
    stream_diff: np.ndarray = None

    # data: List[List[complex]] = field(default_factory=list)

    def to_print(self):
        dt = datetime.datetime.fromtimestamp(self.timestamp // 1000000000)
        print(dt.strftime('%Y-%m-%d %H:%M:%S'))
        print(self.mac, self.nrx, self.ntx, self.num_tones, self.freq)
    # print(self.data)


def read_csi(f, offset, max_csi_frames: int = None, skipped_frames: int = None):
    csi_frames = []
    while max_csi_frames is None or len(csi_frames) < max_csi_frames:
        f.seek(offset, 0)
        # print("test4", offset)
        tmp = f.read(1)
        # print("test1", tmp)
        offset = f.tell()
        if tmp:
            # print("test3", tmp)
            if tmp == pattern[0:1]:
                offset = f.tell()
                tmp = f.read(len(pattern) - 1)
                # print(type(tmp), type(pattern[1:]))
                if tmp == pattern[1:]:

                    # print("test2", tmp)

                    csi = CSI()

                    # timestamp, u64
                    csi.timestamp = struct.unpack('Q', f.read(8))[0]
                    hdr_len = struct.unpack('I', f.read(size_of_int))[0]

                    if hdr_len != 272:
                        continue

                    hdr = f.read(hdr_len)

                    num_csi = struct.unpack('I', f.read(size_of_int))[0]
                    data_len = struct.unpack('I', f.read(size_of_int))[0]
                    csi.num_tones = int(data_len / num_csi / 4)


                    payload_len = num_csi * size_of_int + data_len
                    offset = f.tell() + payload_len

                    if csi.num_tones != 114 or num_csi <= 1:
                        continue

                    if skipped_frames is not None and skipped_frames > 0:
                        skipped_frames = skipped_frames - 1
                        continue

                    # if data_len != struct.unpack('H', hdr[0:2])[0] or csi.num_tones != struct.unpack('H', hdr[52:54])[0]:
                    # continue

                    csi.nrx = num_csi if num_csi < num_antennas else num_antennas
                    csi.ntx = math.ceil(num_csi / num_antennas)

                    csi.mac = hdr[68:74].hex(':')
                    # print(csi.mac)

                    if macs_filter and csi.mac not in macs_filter:
                        continue

                    csi.ftmClock = struct.unpack('I', hdr[8:12])[0] * 3.125
                    csi.muClock = struct.unpack('I', hdr[88:92])[0]
                    csi.freq = struct.unpack('I', hdr[260:264])[0]

                    phase_shift_gen_scidx = 0
                    phase_shift = 0

                    if csi.num_tones == sc_count_legacy:
                        scidx_csi = scidx_legacy_csi
                        scidx_csi_no_pilot = scidx_legacy_csi
                        gen_scidx_csi_no_pilot = ax200_gen_scidx_legacy_csi
                    elif csi.num_tones == sc_count_20mhz:
                        scidx_csi = scidx_20mhz_csi
                        scidx_csi_no_pilot = scidx_20mhz_csi_no_pilot
                        gen_scidx_csi_no_pilot = gen_scidx_20mhz_csi_no_pilot
                    elif csi.num_tones == sc_count_40mhz:
                        scidx_csi = scidx_40mhz_csi
                        scidx_csi_no_pilot = scidx_40mhz_csi_no_pilot
                        gen_scidx_csi_no_pilot = gen_scidx_40mhz_csi_no_pilot
                        phase_shift_gen_scidx = phase_shift_gen_scidx_40mhz
                        phase_shift = phase_shift_40mhz
                    elif csi.num_tones == sc_count_80mhz:
                        scidx_csi = scidx_80mhz_csi
                        scidx_csi_no_pilot = scidx_80mhz_csi_no_pilot
                        gen_scidx_csi_no_pilot = gen_scidx_80mhz_csi_no_pilot
                        phase_shift_gen_scidx = phase_shift_gen_scidx_80mhz
                        phase_shift = phase_shift_80mhz
                    else:  # csi.num_tones == 498:
                        scidx_csi = range(0, csi.num_tones)
                        scidx_csi_no_pilot = range(0, csi.num_tones)
                        gen_scidx_csi_no_pilot = range(0, csi.num_tones)

                    data = f.read(payload_len)

                    # csi.to_print()

                    csi.data = np.zeros((csi.nrx, csi.ntx, csi.num_tones), dtype=complex)
                    csi.phases = np.zeros((csi.nrx, csi.ntx, len(scidx_csi_no_pilot)), dtype=float)
                    csi.amplitude = np.zeros((csi.nrx, csi.ntx, len(scidx_csi_no_pilot)), dtype=float)
                    csi.calib_phases = np.zeros((csi.nrx, csi.ntx, len(scidx_csi_no_pilot)), dtype=float)

                    pos = 0
                    shift_vector = np.zeros(csi.num_tones, dtype=float)
                    shift_vector[phase_shift_gen_scidx:] = phase_shift

                    for i in range(csi.nrx):
                        for j in range(csi.ntx):
                            csi_len = struct.unpack('I', data[pos:pos + size_of_int])[0]
                            pos = pos + size_of_int

                            csi_data = np.array([x[0] for x in struct.iter_unpack('h', data[pos:pos + csi_len])])
                            pos = pos + csi_len

                            csi_data = csi_data[::2] * 1j + csi_data[1::2]
                            csi.data[i][j] = csi_data

                            # get phases
                            phases = np.angle(csi_data)
                            amplitude = np.absolute(csi_data)

                            # remove phase shifts
                            # print(shift_vector)
                            phases = wrap2pi(phases - shift_vector)
                            phases_no_pilot = [phases[x] for x in gen_scidx_csi_no_pilot]
                            amplitude_no_pilot = [amplitude[x] for x in gen_scidx_csi_no_pilot]
                            # plt.plot(phases_no_pilot)
                            # plt.show()
                            csi.phases[i][j] = phases_no_pilot
                            csi.amplitude[i][j] = amplitude_no_pilot

                    # print(num_csi, csi.num_tones)
                    csi_frames.append(csi)
            else:
                # Print if not the beginning of CSI frame
                print(tmp)
        else:
            # Break if EoF
            break
    return csi_frames


def twos_complement(hexstr, bits):
    value = int(hexstr, 16)
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def wrap2pi(phases):
    phases = (phases + np.pi) % (2 * np.pi) - np.pi
    return phases


def remove_slope(phases, scidx_data, slope2remove: float = None):
    tmp_phases = phases
    steps = 0
    if slope2remove is None:
        slope = 0
        unwrap_phases = np.unwrap(tmp_phases)
        while abs(max(unwrap_phases) - min(unwrap_phases)) > np.pi and slope < max_slope:
            steps = steps + 1
            slope = slope + calib_step
            tmp_phases = wrap2pi(tmp_phases - 2 * np.pi * scidx_data * freq_delta * calib_step)
            unwrap_phases = np.unwrap(tmp_phases)
        if slope >= max_slope:
            tmp_phases = phases
            slope = 0
        unwrap_phases = np.unwrap(tmp_phases)
        p = np.polyfit(scidx_data, unwrap_phases, 1)
        slope = slope + p[0] / (2 * np.pi * freq_delta)
        tmp_phases = wrap2pi(tmp_phases + p[0] * scidx_data)
    elif slope2remove == 0:
        unwrap_phases = np.unwrap(tmp_phases)
        p = np.polyfit(scidx_data, unwrap_phases, 1)
        slope = p[0] / (2 * np.pi * freq_delta)
        tmp_phases = wrap2pi(tmp_phases - p[0] * scidx_data)
    else:
        slope = slope2remove
        tmp_phases = wrap2pi(tmp_phases - 2 * np.pi * scidx_data * freq_delta * slope)
    return steps, slope, tmp_phases

def csi_analyzer(path):

    with open(path, "rb") as f:
        csi_frames = read_csi(f, 0, 2000, 0)
        # print(len(csi_frames))
        # print(csi_frames[1].amplitude[0,0,:])
        sub_num_a = np.shape(csi_frames[1].amplitude[0,0,:])[0]
        sub_num_p = np.shape(csi_frames[1].phases[0,0,:])[0]
        csi_amplitude = np.zeros((len(csi_frames), sub_num_a))
        csi_phase = np.zeros((len(csi_frames), sub_num_p))

        if csi_frames:
            for i in range(len(csi_frames)):
                # plt.figure(1)
                # plt.plot(scidx_40mhz_csi_no_pilot, csi_frames[index].phases[0, 0, :])
                # plt.plot(scidx_40mhz_csi_no_pilot, csi_frames[i].amplitude[0, 0, :])
                # plt.show()
                csi_amplitude[i,:] = csi_frames[i].amplitude[0,0,:]
            # plt.show()
    
            for i in range(len(csi_frames)):
                # plt.figure(1)
                # plt.plot(scidx_40mhz_csi_no_pilot, csi_frames[index].phases[0, 0, :])
                # plt.plot(scidx_40mhz_csi_no_pilot, csi_frames[index].amplitude[0, 0, :])
                # plt.show()
                csi_phase[i,:] = csi_frames[i].phases[0,0,:]

    no_frames = np.shape(csi_amplitude)[0]
    no_sub = np.shape(csi_amplitude)[1]
    csi_amplitude_subch = csi_amplitude.copy()
    # csi_amplitude_subch = csi_amplitude_subch[:,12:244]
    # csi_amplitude_subch_buff = np.empty((no_frames, no_subcarriers))
    csi_amplitude_buff = np.zeros(np.shape(csi_amplitude_subch))

    for i in range(no_frames):
    #     csi_amplitude_buff[i,:] = remove_fill(csi_amplitude_subch[i,:],no_pilot)
    #     for j in range(6):
    #         # csi_phase_subch[i,:] = remove_fill(csi_phase_subch[i,:],no_pilot)
    #         csi_amplitude_buff[i,:] = remove_fill(csi_amplitude_buff[i,:],no_dc)
        csi_amplitude_buff[i,:] = running_mean(csi_amplitude_subch[i,:],2)

    csi_amplitude_freq = csi_amplitude_buff.copy()

    scoretable_freq = np.empty((no_frames,1))
    scoretable_freq_std = np.empty((no_frames,1))
    csi_amplitude_grad = np.empty((no_frames,no_sub))

    for i in range(no_frames):
        csi_amplitude_grad[i,:] = np.gradient(csi_amplitude_buff[i,:])

    scoretable_amplitude_grad_std = np.zeros((no_frames,1))

    for i in range(no_frames):
        std_score = 0
        std_score = np.std(csi_amplitude_grad[i,:])
        scoretable_amplitude_grad_std[i,:] = std_score

    scoretable_amplitude_grad_std_sorted = np.argsort(scoretable_amplitude_grad_std, axis=0)
    perc = 0.8
    frame_amplitude_std_selected = scoretable_amplitude_grad_std_sorted[0:int(perc*no_frames),:]

    csi_amplitude_freq_normalized = np.empty((no_frames,np.shape(csi_amplitude_freq)[1]))
    csi_amplitude_freq = remove_nan(csi_amplitude_freq)
    csi_amplitude_freq = remove_nan(csi_amplitude_freq)

    for i in range(no_frames):
        csi_amplitude_freq_normalized[i,:] = preprocessing.normalize([csi_amplitude_freq[i,:]],norm='max')

    csi_amplitude_freq_output = []
    for i in range(no_frames):
        # if i in frame_selected and i in frame_std_selected:
        if i in frame_amplitude_std_selected:
            csi_amplitude_freq_output.append(csi_amplitude_freq_normalized[i,:])
    csi_amplitude_freq_output = np.array(csi_amplitude_freq_output)
    index1 = ['0' for _ in range(np.shape(csi_amplitude_freq_output)[0])]

    # # User profile reading
    # input_path = '/Users/liangxintai/Desktop/CSI_authentication/lab_s1/data_sample/final_User_profile_amplitude/'
    # train_file_name = 'User0.csv'

    # df = pd.read_csv(input_path + train_file_name, index_col=0, header=None)
    # df.reset_index(inplace=True)
    # df = df.reindex(np.random.permutation(df.index))

    df = pd.DataFrame(csi_amplitude_freq_output, index=index1)
    # df.reset_index(inplace=True)
    # df = df.reindex(np.random.permutation(df.index))
    # print('yes')
    return df, csi_amplitude_freq_output

def remove_nan(csi_matrix):
    nan = np.where(np.isnan(csi_matrix))
    nan_line = nan[0]
    nan_list = nan_line.tolist()
    result = [] 

    for i in nan_list: 
        if i not in result: 
            result.append(i) 

    for i in result:
        csi_matrix[i,:] = csi_matrix[i+1,:]

    return csi_matrix

def train_process(df_train):

    df_train.reset_index(inplace=True)
    df_train = df_train.reindex(np.random.permutation(df_train.index))
    x = df_train.iloc[:,1:]
    train_size = int(len(x)*0.8)

    x_train = df_train.iloc[:train_size,1:]
    return x_train

def test_process(df_test):

    df_test.reset_index(inplace=True)
    df_test = df_test.reindex(np.random.permutation(df_test.index))
    test_size = 100
    x_test = df_test.iloc[:test_size,1:]
    x_test = np.array(x_test)

    return x_test

def gen_authenticator(x_train):
    
    ocsvm = OneClassSVM(kernel='rbf',gamma='scale', nu=0.01)
    ocsvm.fit(x_train)

    return ocsvm

def authenticate(ocsvm,x_test_single):

    mid = x_test_single.reshape(1, -1)
    out_test = ocsvm.predict(mid)
    out_test = out_test[0]

    if out_test == 1:
        return b'1'
    elif out_test == -1:
        return b'0'


if __name__ == "__main__":

    df, csi_amplitude_processed = csi_analyzer(path='/Users/liangxintai/Desktop/csi.dat')
    
    for i in range(np.shape(csi_amplitude_processed)[0]):
        plt.plot(scidx_40mhz_csi_no_pilot,csi_amplitude_processed[i,:])
    plt.show()

    print(csi_amplitude_processed)
            
    

        
                

