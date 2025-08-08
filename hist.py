import argparse
import numpy as np
import h5py
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from config import *

psr = argparse.ArgumentParser()
psr.add_argument("-g", dest="geo", type=str, help="geometry file")
psr.add_argument("-i", dest="ipt", nargs="+", type=str, help="input h5 files")  # 支持多个输入文件
psr.add_argument("-o", dest="opt", nargs="+", type=str, help="output parquet files")  # 输出两个 parquet 文件
psr.add_argument("-b", dest="Bins", type=int, help="number of spatial bins")
psr.add_argument("-t", dest="T_Bins", type=int, help="number of timing bins")
args = psr.parse_args()

# 读取PMT几何数据
with h5py.File(args.geo, 'r') as h5file_r:
    geo = h5file_r["Geometry"][...]
geo_theta = np.deg2rad(geo["theta"][:])

# 计算分bin
y_bins = np.arange(args.Bins+1) / args.Bins
x_bins = np.concatenate((-y_bins[::-1], y_bins[1:]))
t_bins = np.arange(int(args.T_Bins)+1) / int(args.T_Bins) * T_MAX

# 累加所有输入文件
nev, npe, x_sum, y_sum = 0, 0, 0, 0
for fname in tqdm(args.ipt):
    fname_base = os.path.basename(fname)
    z_val = float(os.path.splitext(fname_base)[0])
    with h5py.File(fname, 'r') as h5file_r:
        PETruth = h5file_r["PETruth"][...]
    z = z_val / R0
    r = np.full((N,), z, dtype='float32')
    θ = geo_theta
    px = r * np.cos(θ)
    py = r * np.sin(θ)
    x_sum += np.histogram2d(px, py, bins=[x_bins, y_bins], weights=px)[0]
    y_sum += np.histogram2d(px, py, bins=[x_bins, y_bins], weights=py)[0]
    nev += np.histogram2d(px, py, bins=[x_bins, y_bins])[0]
    npe += np.histogramdd(([px[PETruth['ChannelID']], py[PETruth['ChannelID']], PETruth['PETime']]), bins=[x_bins, y_bins, t_bins])[0]

nev_t = np.repeat(nev[:, :, np.newaxis], args.T_Bins, axis=2)
use_s = nev > 0
use_t = nev_t > 0

nev_s = nev[use_s]
x_sum_s = x_sum[use_s]
y_sum_s = y_sum[use_s]
x_mean_s = x_sum_s / nev_s
y_mean_s = y_sum_s / nev_s
npe_s = np.sum(npe, axis=2)[use_s]

nev_t = nev_t[use_t]
npe_t = npe[use_t]
x_sum_t = np.repeat(x_sum[:, :, np.newaxis], args.T_Bins, axis=2)[use_t]
y_sum_t = np.repeat(y_sum[:, :, np.newaxis], args.T_Bins, axis=2)[use_t]
x_mean_t = x_sum_t / nev_t
y_mean_t = y_sum_t / nev_t

pt_ini = (t_bins[1:] + t_bins[:-1]) / 2
t_mean = np.tile(pt_ini, (args.Bins*2, args.Bins, 1))[use_t]

table_s = pa.Table.from_arrays([x_mean_s.astype('float32'), y_mean_s.astype('float32'),
                                nev_s.astype('uint32'), npe_s.astype('uint32')],
    names=["x", "y", "nEV", "nPE"])

table_t = pa.Table.from_arrays([x_mean_t.astype('float32'), y_mean_t.astype('float32'), t_mean.astype("float32"), 
                                nev_t.astype('uint32'), npe_t.astype('uint32')],
    names=["x", "y", "t", "nEV", "nPE"])

pq.write_table(table_s, args.opt[0], compression="ZSTD")
pq.write_table(table_t, args.opt[1], compression="ZSTD")


