import os
import time
from pandas import read_csv
from multiprocessing import Pool

import torch

import numpy as np
import numpy.typing as npt

import bnnc

def parse_predictions(c_prediction_path):
    df = read_csv(c_prediction_path)
    bgm = []
    mc_samples = df["mcpass"].max() + 1
    for i in range(mc_samples):
        bgm.append(df[(df["mcpass"] == i)].filter(regex="class").values)
    return np.array(bgm)

def parallel_c(x: tuple[int, npt.NDArray, bnnc.model_info.ModelInfo]):
    i, data, model_info = x
    worker_folder = f"Code/worker_{i}"

    os.system(f"""
        rm -rf {worker_folder}
        mkdir -p {worker_folder}
        cp -r {bnnc.model_info.c_sources_abspath}/bnn {worker_folder}
        cp {bnnc.model_info.c_sources_abspath}/Makefile {worker_folder}
        cp {bnnc.model_info.c_sources_abspath}/test_main.c {worker_folder}/main.c
        cp Code/bnn_config.h {worker_folder}
        cp Code/bnn_model.h {worker_folder}
        cp Code/bnn_model_weights.h {worker_folder}
    """)

    d = model_info.create_c_data(data)
    with open(f"{worker_folder}/test_data.h", "w") as f:
        f.write(d)

    os.system(f"""
        cd {worker_folder}
        make main > run.log
    """)

    return parse_predictions(f"{worker_folder}/run.log")

def run_c_model(model_info, test_data, num_workers, max_img_per_worker):

    l, h, w = model_info.create_c_code()
    with open("Code/bnn_config.h", "w") as f:
        f.write(l)
    with open("Code/bnn_model.h", "w") as f:
        f.write(h)
    with open("Code/bnn_model_weights.h", "w") as f:
        f.write(w)

    num_targets = num_workers * max_img_per_worker
    split_data = np.split(test_data[:num_targets], num_workers)

    model_info.print_cinfo()

    with Pool(num_workers) as p:
        work = []
        for i, data in enumerate(split_data):
            work.append((i+1, data, model_info))

        time_start = time.time()
        print(f"{time.strftime("%H:%M:%S", time.localtime(time_start))} -- Starting C predictions {num_targets} targets")
        preds = np.concatenate(p.map(parallel_c, work), 1)
        preds[preds < 0] = 0
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - time_start))
        print(f"{num_targets} img C preditions done in {elapsed_time} using {num_workers} threads")
        np.savez("Code/predictions", preds)

import testconf

class CrunParams:
    # C config
    num_workers = 20
    max_img_per_worker = 500


def eval_model(modelname:str, generation_method:str, fixed_bits:int):
    print(f"Model: {modelname}")

    _, test_data = testconf.get_data(modelname)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
    for data, targets in test_loader:
        pass
    data = data.permute((0,2,3,1)).detach().numpy()
    targets = targets.detach().numpy()
    input_shape = np.array(data[0].shape)
    flat_data = data.reshape((data.shape[0], -1))

    model, _ = testconf.get_model(modelname, "bnn")

    model_info = bnnc.torch.info_from_model(model, "bnn_model")
    model_info.calculate_buffers(input_shape)
    #model_info.print_buffer_info()

    if generation_method == "uniform":
        model_info.uniform_weight_transform()
    elif generation_method == "bernoulli":
        model_info.bernoulli_weight_transform()
    elif generation_method == "gaussian":
        pass

    model_info.fixed_bits = fixed_bits
    run_c_model(model_info, flat_data, CrunParams.num_workers, CrunParams.max_img_per_worker)
    os.system(f"mv Code/predictions.npz {testconf.prediction_path(modelname, generation_method, fixed_bits)}")

    preds = np.load(testconf.prediction_path(modelname, generation_method, fixed_bits))["arr_0"]

    targets = np.array(test_data.targets)[:preds.shape[1]]

    cdata = (*bnnc.uncertainty.analyze_predictions(preds, targets), preds)
    cacc = bnnc.uncertainty.accuracy(cdata[0])
    cece, cuce = bnnc.uncertainty.calibration_errors(cdata[0])

    print("ACC %", cacc * 100)
    print("ECE %", cece * 100)
    print("UCE %", cuce * 100)

if __name__ == "__main__":
    testconf.init_folders()

    for model in testconf.Conf.model_list:
        eval_model(model, "gaussian", 10)
        print('')
