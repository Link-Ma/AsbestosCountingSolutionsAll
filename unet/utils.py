from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import glob
import gzip
from itertools import islice
import functools
from pathlib import Path
from pprint import pprint
import random
import shutil
from typing import Dict, List, Tuple
import warnings

import matplotlib.pyplot as plt

import cv2
import json_lines
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.affinity import rotate
import skimage.io
import skimage.exposure
from skimage import exposure
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
import statprof
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
# image preprocessing on PIL: PIL to Tensor, normalize image, compose several transforms together
from torchvision.transforms import ToTensor, Normalize, Compose
import tqdm


N_CLASSES = 1

# test if cuda is available
cuda_is_available = torch.cuda.is_available()

# if x is list or tuple, if yes iter it. if not cuda it.
def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))

# compatible for cpu and gpu 
def cuda(x):
    return x.cuda() if cuda_is_available else x

# image preprocessing, see import
img_transform = Compose([
    ToTensor()
    # why set these values
#     Normalize(mean=[0.44, 0.46, 0.46], std=[0.16, 0.15, 0.15]),
])

# timer
def profile(fn):
    # decorator
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        statprof.start()
        try:
            return fn(*args, **kwargs)
        finally:
            statprof.stop()
            statprof.display()
    return wrapped

# annotation return ndarray
# load arrays or objects from npy or npz or pickled file
def load_image(path: Path, *, cache: bool) -> np.ndarray:
    cached_path = path.parent / 'cache' / (path.stem + '.npy')  # type: Path
    if cache and cached_path.exists():
        return np.load(str(cached_path))
    # cv2. imread vs PIL read
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = increase_img_contrast(img)
#     if path.parent.name == 'Train':
#         # mask with TrainDotted
#         img_dotted = cv2.imread(str(path.parent.parent / 'TrainDotted' / path.name))
#         # what is axis2
#         img_dotted = cv2.cvtColor(img_dotted, cv2.COLOR_BGR2RGB)
#         # why set row to 0
        #img[img_dotted.sum(axis=2) == 0, :] = 0
    # save processed image if cache
    # wb for image files(binary) like jpg or exe
    if cache:
        with cached_path.open('wb') as f:
            np.save(f, img)
    return img

def increase_img_contrast(img: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img, (2, 98))
    r = exposure.rescale_intensity(img, in_range=(p2, p98))
    return r

# why divide by 1000
def load_pred(path: Path) -> np.ndarray:
    with gzip.open(str(path), 'rb') as f:
        return np.load(f).astype(np.float32) / 1000

# get images without mismatched ones
def labeled_paths() -> List[Path]:
    # https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/30895
    # mismatched = pd.read_csv(str(DATA_ROOT / 'MismatchedTrainImages.txt'))
    # bad_ids = set(mismatched.train_id)
    # # https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/31424
    # # add 941 and 200 to bad ids 
    # bad_ids.update([941,  200])
    # # FIXME - these are valid but have no coords, get them (esp. 912)!
    # # https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/31472#175541
    # bad_ids.update([491, 912])
    return [p for p in DATA_ROOT.joinpath('Train').glob('*.png')]

# resize images
def downsample(img: np.ndarray, ratio: int=4) -> np.ndarray:
    h, w = img.shape
    h = int(h / ratio)
    w = int(w / ratio)
    return cv2.resize(img, (w, h))

# 
def make_loader(dataset_cls: type,
                args, paths: List[Path], coords: pd.DataFrame,
                deterministic: bool=False, **kwargs) -> DataLoader:
    dataset = dataset_cls(
        img_paths=paths,
        # with label
        coords=coords,
        #patch size
        size=args.patch_size,
        transform=img_transform,
        deterministic=deterministic,
        **kwargs,
    )
    return DataLoader(
        dataset=dataset,
        shuffle=True,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )

# write log 
def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()

# When a module is loaded in Python, __file__ is set to its name
DATA_ROOT = Path(__file__).absolute().parent / 'data'

# what is stratified: Stratification is the process of rearranging the data as to ensure each fold is a good representative of the whole.
def train_valid_split(args) -> Tuple[List[Path], List[Path]]:
    img_paths = labeled_paths()
    # if args.limit and len(img_paths) > args.limit:
    #     random.seed(42)
    #     img_paths = random.sample(img_paths, args.limit)
    # if args.stratified:
    #     sorted_ids = coords['cls'].groupby(level=0).count().sort_values().index
    #     idx_by_id = {img_id: idx for idx, img_id in enumerate(sorted_ids)}
    #     img_paths.sort(key=lambda p: idx_by_id.get(int(p.stem), len(sorted_ids)))
    #     train, test = [], []
    #     for i, p in enumerate(img_paths):
    #         if i % args.n_folds == args.fold - 1:
    #             test.append(p)
    #             train.append(p)
    #     return train, test
    # else:
    img_paths = np.array(sorted(img_paths))
    cv_split = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    # split data into train and test, return ndarray
    img_folds = list(cv_split.split(img_paths))
    print("img_folds: ", img_folds)
    train_ids, valid_ids = img_folds[args.fold - 1]
    print("train ", train_ids, "valid ", valid_ids)
    return img_paths[train_ids], img_paths[valid_ids]


def load_coords():
    return pd.read_csv(str(DATA_ROOT / 'coords-threeplusone-v0.4.csv'),
                       index_col=0)


class BaseDataset(Dataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 ):
        # load images
        self.img_ids = [int(p.name.split('.')[0]) for p in img_paths]
        self.imgs = {img_id: load_image(p, cache=False)
                     for img_id, p in tqdm.tqdm(list(zip(self.img_ids, img_paths)),
                                                desc='Images')}
        self.imgs_iter = []
        for i in self.imgs.items():
            self.imgs_iter.append(i)
        # get related coords
        self.coords = coords.loc[self.img_ids].dropna()
        self.coords_by_img_id = {}
#         print(self.coords)
        for img_id in self.img_ids:
            try:
                coords = self.coords.loc[[img_id]]
            except KeyError:
                coords = []
            self.coords_by_img_id[img_id] = coords

# core!
class BasePatchDataset(BaseDataset):
    def __init__(self,
                 img_paths: List[Path],
                 coords: pd.DataFrame,
                 transform,
                 size: int,
                 min_scale: float=1.,
                 max_scale: float=1.,
                 oversample: float=0.,
                 deterministic: bool=False,
                 ):
        super().__init__(img_paths, coords)
        self.patch_size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.oversample = oversample
        self.transform = transform
        self.deterministic = deterministic
        #self.first_part = {}

    def __getitem__(self, idx):
        while True:
            pp = self.get_patch_points(idx)
            if pp is not None:
                return self.new_x_y(*pp)

    def new_x_y(self, patch, points, idx):
        """ Sample (x, y) pair.
        """
        raise NotImplementedError

    def get_patch_points(self, idx):
        lag = 0
        if idx > len(self.imgs) - 1:
            lag = 900
        idx = idx % len(self.imgs)
            
        img = self.imgs_iter[idx][1]
        max_y, max_x = img.shape[:2]
#         print(max_y, max_x)
        patch = img[:, lag : (lag + max_y)]
        
        coords = self.coords_by_img_id[self.imgs_iter[idx][0]]
        patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        points = []
        if len(coords) > 0:
            for cls, col, row in zip(coords.cls, coords.col, coords.row):
                if lag == 900 and row >= 900:
                    points.append((cls, (row - lag, col)))
                if lag == 0 and row <= 1800 :
                    points.append((cls, (row, col)))
#         print("get_patch_points function return")
        return patch, points, self.imgs_iter[idx][0]+lag

    def __len__(self):
        return len(self.imgs) * 2

# what is lr: learning rate
def train(args, model: nn.Module, criterion, *, train_loader, valid_loader,
          make_optimizer=None, save_predictions, is_classification=False):
    lr = args.lr
    make_optimizer = make_optimizer or (lambda lr: Adam(model.parameters(), lr=lr))
    optimizer = make_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model.pt'
    best_model_path = root / 'best-model.pt'
    if model_path.exists():
        # Loads an object saved with torch.save() from a file.
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')

    def save(ep: int):
        torch.save(
            {'model': model.state_dict(),
             'epoch': ep,
             'step': step,
             'best_valid_loss': best_valid_loss,
             }, str(model_path))
        shutil.copy(str(model_path), str(root / 'model-{}.pt'.format(ep)))

    report_each = 10
    save_prediction_each = report_each * 10
    root = Path(args.root)
    # a for write, t for text
    log = root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
#     print("trainLoader:", len(train_loader))
    for epoch in range(epoch, args.n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
                
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                (batch_size * loss).backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.3f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    if save_predictions and i % save_prediction_each == 0:
                        p_i = (i // save_prediction_each) % 5
                        save_predictions(root, p_i, inputs, targets, outputs)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader,
                                       is_classification=is_classification)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif len(valid_losses) > 2 and min(valid_losses[-2:]) > best_valid_loss:
                # two epochs without improvement
                lr /= 5
                optimizer = make_optimizer(lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def rotated(patch, angle):
    size = patch.shape[:2]
    center = tuple(np.array(size) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(patch, rot_mat, size, flags=cv2.INTER_LINEAR)


def save_image(fname, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        skimage.io.imsave(fname, data)


CLS_COLORS = [
    [1., 0., 0.],  # red: adult males
]
CLS_NAMES = ['0']


def validation(model: nn.Module, criterion, valid_loader,
               is_classification=False) -> Dict[str, float]:
    # set the module to evaluation mode
    model.eval()
    losses = []
    all_targets, all_outputs = [], []
    for inputs, targets in valid_loader:
        inputs, targets = variable(inputs, volatile=True), variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        if is_classification:
            all_targets.extend(targets.data.cpu().numpy())
            all_outputs.extend(outputs.data.cpu().numpy().argmax(axis=1))
    valid_loss = np.mean(losses)  # type: float
    metrics = {'valid_loss': valid_loss}
    print('Valid loss: {:.5f}'.format(valid_loss))
    if is_classification:
        accuracy = accuracy_score(all_targets, all_outputs)
        print('Accuracy: {:.3f}'.format(accuracy))
        print(classification_report(all_targets, all_outputs))
        metrics['accuracy'] = accuracy
    return metrics


def load_best_model(model: nn.Module, root: Path, model_path=None) -> None:
    model_path = model_path or str(root / 'best-model.pt')
    state = torch.load(model_path)
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))


def batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def imap_fixed_output_buffer(fn, it, threads: int):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        max_futures = threads + 1
        for x in it:
            while len(futures) >= max_futures:
                future, futures = futures[0], futures[1:]
                yield future.result()
            futures.append(executor.submit(fn, x))
        for future in futures:
            yield future.result()


def plot(*args, ymin=None, ymax=None, xmin=None, xmax=None, params=False,
         max_points=200):
    """ Use in the notebook like this:
    plot('./runs/oc2', './runs/oc1', 'loss', 'valid_loss')
    """
    paths, keys = [], []
    for x in args:
        if x.startswith('.') or x.startswith('/'):
            if '*' in x:
                paths.extend(glob.glob(x))
            else:
                paths.append(x)
        else:
            keys.append(x)
    plt.figure(figsize=(12, 8))
    keys = keys or ['loss', 'valid_loss']

    ylim_kw = {}
    if ymin is not None:
        ylim_kw['ymin'] = ymin
    if ymax is not None:
        ylim_kw['ymax'] = ymax
    if ylim_kw:
        plt.ylim(**ylim_kw)

    xlim_kw = {}
    if xmin is not None:
        xlim_kw['xmin'] = xmin
    if xmax is not None:
        xlim_kw['xmax'] = xmax
    if xlim_kw:
        plt.xlim(**xlim_kw)
    for path in paths:
        path = Path(path)
        with json_lines.open(str(path.joinpath('train.log')), broken=True) as f:
            events = list(f)
        if params:
            print(path)
            pprint(json.loads(path.joinpath('params.json').read_text()))
        for key in sorted(keys):
            xs, ys = [], []
            for e in events:
                if key in e:
                    xs.append(e['step'])
                    ys.append(e[key])
            if xs:
                if len(xs) > 2 * max_points:
                    indices = (np.arange(0, len(xs), len(xs) / max_points)
                               .astype(np.int32))
                    xs = np.array(xs)[indices[1:]]
                    ys = [np.mean(ys[idx: indices[i + 1]])
                          for i, idx in enumerate(indices[:-1])]
                plt.plot(xs, ys, label='{}: {}'.format(path, key))
    plt.legend()