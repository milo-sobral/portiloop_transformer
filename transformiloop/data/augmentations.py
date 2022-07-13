import numpy as np
import torch


def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    # print(type(b))
    return b.astype(int)

def DataTransform(sample, aug_config):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, aug_config['jitter_scale_ratio'])
    # weak_aug = permutation(sample, max_segments=config.augmentation.max_seg)
    strong_aug = jitter(permutation(sample, max_segments=aug_config['max_seg']), aug_config['jitter_ratio'])

    return weak_aug, strong_aug

def DataTransform_TD(sample, aug_config):
    """Weak and strong augmentations"""
    aug_1 = jitter(sample, aug_config['jitter_ratio'])
    aug_2 = scaling(sample, aug_config['jitter_scale_ratio'])
    if sample.shape[0] > 1:
        aug_3 = permutation(sample, max_segments=aug_config['max_seg'])
    # li = np.random.randint(0, 4, size=[sample.shape[0]])
    # li_onehot = one_hot_encoding(li)
    # aug_1[1-li_onehot[:, 0]] = 0 # the rows are not selected are set as zero.
    # aug_2[1 - li_onehot[:, 1]] = 0
    aug_T = aug_1 + aug_2
    if sample.shape[0] > 1:
    #     aug_3[1 - li_onehot[:, 2]] = 0
        aug_T += aug_3 #+aug_4
    # print('got here 2')
    
    return aug_T


def DataTransform_FD(sample):
    """Weak and strong augmentations in Frequency domain """
    aug_1 =  remove_frequency(sample, 0.1)
    aug_2 = add_frequency(sample, 0.1)

    # generate random sequence
    # li = np.random.randint(0, 2, size=[sample.shape[0]]) # there are two augmentations in Frequency domain
    # li_onehot = one_hot_encoding(li)
    # aug_1[1-li_onehot[:, 0]] = 0 # the rows are not selected are set as zero.
    # aug_2[1 - li_onehot[:, 1]] = 0
    aug_F = aug_1 + aug_2
    return aug_F



def generate_binomial_mask(B, T, D, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)

def masking(x, mask='binomial'):
    nan_mask = ~x.isnan().any(axis=-1)
    x[~nan_mask] = 0

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=0.9).to(x.device)

    # mask &= nan_mask
    x[~mask_id] = 0
    return x

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def remove_frequency(x, maskout_ratio=0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > maskout_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0,):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix