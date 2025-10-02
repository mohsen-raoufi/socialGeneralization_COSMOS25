# model sims with CL model
import numpy as np
import json
import os

import modelSim as ms
path = "./environments_unequal"
json_files = [file for file in os.listdir(path) if file.endswith('.json')]
EnvList = []
for file in json_files:
    f=open(os.path.join(path, file))
    EnvList.append(json.load(f))

allParams = ms.param_gen(4, 10, [6, 0, 0, 0])
rounds = 20
shor = 40
result = ms.model_sim(allParams, EnvList, rounds, shor)
result.to_csv("cl_model_sims.csv", index=False)
