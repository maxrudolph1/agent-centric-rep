import h5py

filename = '/u/mrudolph/documents/acro/vd4rl/cheetah_run/500k/cheetah_run_different_video_500K.hdf5'

with h5py.File(filename, 'r') as f:
    print(f.keys())
    print(f['observation'].shape)
    print(f['action'].shape)
    print(f['reward'].shape)
    print(f['discount'].shape)
    import pdb; pdb.set_trace()
    step_type = f['step_type'][:]
    rewards = f['reward'][:]


import matplotlib.pyplot as plt
plt.hist(rewards, bins=100)
plt.savefig('rewards.png')