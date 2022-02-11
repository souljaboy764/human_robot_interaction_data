from glob import glob
from matplotlib import pyplot as plt
import numpy as np

import torch
from torch.nn import functional as F

from read_hh_hr_data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_actions = {}
new_length = 450
theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device)
train_data = []
test_data = []
for action in ['hand_wave', 'hand_shake', 'rocket', 'parachute']:
	data_actions[action] = []
	for trial in ['1','2']:
		data_file = 'hh/p1/'+action+'_s1_'+trial+'.csv'
		segment_file = 'hh/segmentation/'+action+'_'+trial+'.npy'
		data_p, data_q, times = read_data(data_file)
		segments = np.load(segment_file)
		for s in segments:
			if (s[1]-s[0] >400):
				in_traj = data_p[s[0]:s[1], joints_dic['RightHand']]
				grid = F.affine_grid(theta, torch.Size([1, 3, 2, new_length]), align_corners=True)
				traj = torch.tensor(data_p[s[0]:s[1], joints_dic['RightHand']]).to(device).unsqueeze(0).transpose(1,2).unsqueeze(2)
				traj = torch.concat([traj, torch.zeros_like(traj)], dim=2) # batch, trajectory_dim, 2, trajectory_size
				window = F.grid_sample(traj.type(torch.float32), grid, align_corners=True)[0, :, 0].transpose(0,1).cpu().detach().numpy()
				data_actions[action].append(window)
	train_data += data_actions[action][:26]
	test_data += data_actions[action][26:31]

train_data = np.array(train_data)
test_data = np.array(test_data)
print(train_data.shape, test_data.shape)
np.savez_compressed('buetepage_trainData.npz', train_data)
np.savez_compressed('buetepage_testData.npz', test_data)