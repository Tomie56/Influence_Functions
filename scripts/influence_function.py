import torch
import time
import datetime
import numpy as np
import copy
import logging
from tqdm import tqdm
from pathlib import Path

import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import grad
from torch.autograd.functional import vhp
from torch.utils.data import DataLoader
from scipy.optimize import fmin_ncg

# Device management
def get_device(gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    else:
        return torch.device("cpu")

# Conjugate gradient for Ax - b = 0
def conjugate_gradient(ax_fn, b, debug_callback=None, avextol=None, maxiter=None):
    cg_callback = None
    if debug_callback:
        cg_callback = lambda x: debug_callback(
            x, -np.dot(b, x), 0.5 * np.dot(x, ax_fn(x))
        )

    result = fmin_ncg(
        f=lambda x: 0.5 * np.dot(x, ax_fn(x)) - np.dot(b, x),
        x0=np.zeros_like(b),
        fprime=lambda x: ax_fn(x) - b,
        fhess_p=lambda x, p: ax_fn(p),
        callback=cg_callback,
        avextol=avextol,
        maxiter=maxiter,
    )

    return result

# Attribute manipulation
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

# Functional model conversion
def make_functional(model):
    orig_params = tuple(model.parameters())
    names = []

    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)

    return orig_params, names

def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), Parameter(p))

# Tensor <-> parameter conversion
def tensor_to_tuple(vec, parameters):
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))

    pointer = 0
    split_tensors = []
    for param in parameters:
        num_param = param.numel()
        split_tensors.append(vec[pointer:pointer + num_param].view_as(param))
        pointer += num_param

    return tuple(split_tensors)

def parameters_to_vector(parameters):
    vec = []
    for param in parameters:
        vec.append(param.view(-1))

    return torch.cat(vec)

# s_test via conjugate gradient
def s_test_cg(x_test, y_test, model, train_loader, damp, gpu=-1, verbose=True, loss_func="cross_entropy"):
    device = get_device(gpu)
    x_test, y_test = x_test.to(device), y_test.to(device)

    v_flat = parameters_to_vector(grad_z(x_test, y_test, model, gpu, loss_func=loss_func))

    def hvp_fn(x):
        x_tensor = torch.tensor(x, requires_grad=False).to(device)

        params, names = make_functional(model)
        params = tuple(p.detach().requires_grad_() for p in params)
        flat_params = parameters_to_vector(params)

        hvp = torch.zeros_like(flat_params).to(device)

        for x_train, y_train, train_ids in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)

            def f(flat_params_):
                split_params = tensor_to_tuple(flat_params_, params)
                load_weights(model, names, split_params)
                out = model(x_train)
                loss = calc_loss(out, y_train)
                return loss

            batch_hvp = vhp(f, flat_params, x_tensor, strict=True)[1]
            hvp += batch_hvp / float(len(train_loader))

        with torch.no_grad():
            load_weights(model, names, params, as_params=True)
            damped_hvp = hvp + damp * v_flat

        return damped_hvp.cpu().numpy()

    def print_function_value(_, f_linear, f_quadratic):
        print(
            f"Conjugate function value: {f_linear + f_quadratic}, lin: {f_linear}, quad: {f_quadratic}"
        )

    debug_callback = print_function_value if verbose else None

    result = conjugate_gradient(
        hvp_fn,
        v_flat.cpu().numpy(),
        debug_callback=debug_callback,
        avextol=1e-8,
        maxiter=100,
    )

    result = torch.tensor(result).to(device)
    return result

# Stochastic s_test (IHVP)
def s_test(x_test, y_test, model, i, samples_loader, gpu=-1, damp=0.01, scale=25.0, loss_func="cross_entropy"):
    model.eval()
    device = get_device(gpu)
    x_test, y_test = x_test.to(device), y_test.to(device)

    v = grad_z(x_test, y_test, model, gpu, loss_func=loss_func)
    h_estimate = v

    params, names = make_functional(model)
    params = tuple(p.detach().requires_grad_().to(device) for p in params)

    progress_bar = tqdm(samples_loader, desc=f"IHVP {i}")
    for i, (x_train, y_train, train_ids) in enumerate(progress_bar):
        x_train, y_train = x_train.to(device), y_train.to(device)

        def f(*new_params):
            load_weights(model, names, new_params)
            out = model(x_train)
            loss = calc_loss(out, y_train, loss_func=loss_func)
            return loss

        hv = vhp(f, params, tuple(h_estimate), strict=True)[1]

        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]

            if i % 100 == 0:
                norm = sum([h_.norm() for h_ in h_estimate])
                progress_bar.set_postfix({"est_norm": norm.item()})

    with torch.no_grad():
        load_weights(model, names, params, as_params=True)

    return h_estimate

# Loss calculation
def calc_loss(logits, labels, loss_func="cross_entropy"):
    if loss_func == "cross_entropy":
        if logits.shape[-1] == 1:
            loss = F.binary_cross_entropy_with_logits(logits, labels.type(torch.float))
        else:
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
    elif loss_func == "mean":
        loss = torch.mean(logits)
    else:
        raise ValueError("{} is not a valid value for loss_func".format(loss_func))

    return loss

# Gradient z (model params -> loss)
def grad_z(x, y, model, gpu=-1, loss_func="cross_entropy"):
    model.eval()
    device = get_device(gpu)
    x, y = x.to(device), y.to(device)

    prediction = model(x)
    loss = calc_loss(prediction, y, loss_func=loss_func)

    return grad(loss, model.parameters())

# Single test sample s_test calculation
def s_test_sample(
    model,
    x_test,
    y_test,
    train_loader,
    gpu=-1,
    damp=0.01,
    scale=25,
    recursion_depth=5000,
    r=1,
    loss_func="cross_entropy",
):
    model.eval()
    device = get_device(gpu)
    x_test, y_test = x_test.to(device), y_test.to(device)

    inverse_hvp = [
        torch.zeros_like(params, dtype=torch.float).to(device) for params in model.parameters()
    ]

    for i in range(r):
        hessian_loader = DataLoader(
            train_loader.dataset,
            sampler=torch.utils.data.RandomSampler(
                train_loader.dataset, True, num_samples=recursion_depth
            ),
            batch_size=1,
            collate_fn=train_loader.collate_fn,
        )

        cur_estimate = s_test(
            x_test, y_test, model, i, hessian_loader, gpu=gpu, damp=damp, scale=scale, loss_func=loss_func,
        )

        with torch.no_grad():
            inverse_hvp = [
                old + (cur / scale) for old, cur in zip(inverse_hvp, cur_estimate)
            ]

    with torch.no_grad():
        inverse_hvp = [component / r for component in inverse_hvp]

    return inverse_hvp

# Full test set s_test calculation
def calc_s_test(
    model,
    test_loader,
    train_loader,
    save=False,
    gpu=-1,
    damp=0.01,
    scale=25,
    recursion_depth=5000,
    r=1,
    start=0,
):
    model.eval()
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    if start < 0 or start >= len(test_loader.dataset):
        raise ValueError(f"start index {start} is out of range (0 to {len(test_loader.dataset)-1})")

    s_tests = []
    for i in range(start, len(test_loader.dataset)):
        x_item, y_item, _ = test_loader.dataset[i]
        test_batch = [(x_item, y_item, i)]
        z_test, t_test, _ = test_loader.collate_fn(test_batch)

        s_test_vec = s_test_sample(
            model, z_test, t_test, train_loader, gpu, damp, scale, recursion_depth, r
        )

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec, save.joinpath(f"{i}_recdep{recursion_depth}_r{r}.s_test")
            )
        else:
            s_tests.append(s_test_vec)

    return s_tests, save

# Full train set grad_z calculation
def calc_grad_z(model, train_loader, save_pth=False, gpu=-1, start=0):
    model.eval()
    if save_pth and isinstance(save_pth, str):
        save_pth = Path(save_pth)
    if not save_pth:
        logging.info("ATTENTION: Not saving grad_z files!")

    if start < 0 or start >= len(train_loader.dataset):
        raise ValueError(f"start index {start} is out of range (0 to {len(train_loader.dataset)-1})")

    grad_zs = []
    for i in range(start, len(train_loader.dataset)):
        x_item, y_item, _ = train_loader.dataset[i]
        train_batch = [(x_item, y_item, i)]
        z, t, _ = train_loader.collate_fn(train_batch)
        
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if save_pth:
            grad_z_vec = [g.cpu() for g in grad_z_vec]
            torch.save(grad_z_vec, save_pth.joinpath(f"{i}.grad_z"))
        else:
            grad_zs.append(grad_z_vec)

    return grad_zs, save_pth

# Load precomputed s_test
def load_s_test(
    s_test_dir=Path("./s_test/"), s_test_id=0, r_sample_size=10, train_dataset_size=-1
):
    if isinstance(s_test_dir, str):
        s_test_dir = Path(s_test_dir)

    s_test = []
    logging.info(f"Loading s_test from: {s_test_dir} ...")
    num_s_test_files = len(list(s_test_dir.glob("*.s_test")))
    if num_s_test_files != r_sample_size:
        logging.warning(
            "Load Influence Data: number of s_test sample files"
            " mismatches the available samples"
        )

    for i in range(num_s_test_files):
        try:
            s_test.append(torch.load(s_test_dir / str(s_test_id) + f"_{i}.s_test"))
        except FileNotFoundError:
            logging.warning(f"s_test file {s_test_id}_{i}.s_test not found, skipping...")
            continue

    e_s_test = s_test[0]
    for i in range(len(s_test)):
        e_s_test = [i + j for i, j in zip(e_s_test, s_test[0])]

    e_s_test = [i / len(s_test) for i in e_s_test]

    return e_s_test, s_test

# Load precomputed grad_z
def load_grad_z(grad_z_dir=Path("./grad_z/"), train_dataset_size=-1):
    if isinstance(grad_z_dir, str):
        grad_z_dir = Path(grad_z_dir)

    grad_z_vecs = []
    logging.info(f"Loading grad_z from: {grad_z_dir} ...")
    available_grad_z_files = len(list(grad_z_dir.glob("*.grad_z")))
    if available_grad_z_files != train_dataset_size:
        logging.warning(
            "Load Influence Data: number of grad_z files mismatches" " the dataset size"
        )
        if -1 == train_dataset_size:
            train_dataset_size = available_grad_z_files
    for i in range(train_dataset_size):
        try:
            grad_z_vecs.append(torch.load(grad_z_dir / str(i) + ".grad_z"))
        except FileNotFoundError:
            logging.warning(f"grad_z file {i}.grad_z not found, skipping...")
            continue

    return grad_z_vecs

# Calculate influence function
def calc_influence_function(train_dataset_size, grad_z_vecs=None, e_s_test=None):
    if not grad_z_vecs and not e_s_test:
        grad_z_vecs = load_grad_z()
        e_s_test, _ = load_s_test(train_dataset_size=train_dataset_size)

    if len(grad_z_vecs) != train_dataset_size:
        logging.warning(
            "Training data size and the number of grad_z files are" " inconsistent."
        )
        train_dataset_size = len(grad_z_vecs)

    influences = []
    for i in range(train_dataset_size):
        tmp_influence = (
            -sum(
                [
                    torch.sum(k * j).data.cpu().numpy()
                    for k, j in zip(grad_z_vecs[i], e_s_test)
                ]
            )
            / train_dataset_size
        )
        influences.append(tmp_influence)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()

# Calculate influence for single test sample
def calc_influence_single(
    model,
    train_loader,
    test_loader,
    test_id_num,
    recursion_depth,
    r,
    gpu=0,
    damp=0.01,
    scale=25,
    s_test_vec=None,
    time_logging=False,
    loss_func="cross_entropy",
):
    model.eval()
    device = get_device(gpu)

    if s_test_vec is None:
        x_item, y_item, _ = test_loader.dataset[test_id_num]
        test_batch = [(x_item, y_item, test_id_num)]
        z_test, t_test, _ = test_loader.collate_fn(test_batch)
        
        s_test_vec = s_test_sample(
            model,
            z_test,
            t_test,
            train_loader,
            gpu,
            recursion_depth=recursion_depth,
            r=r,
            damp=damp,
            scale=scale,
            loss_func=loss_func,
        )

    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in tqdm(range(train_dataset_size)):
        x_item, y_item, _ = train_loader.dataset[i]
        train_batch = [(x_item, y_item, i)]
        z, t, _ = train_loader.collate_fn(train_batch)

        if time_logging:
            time_a = datetime.datetime.now()

        grad_z_vec = grad_z(z, t, model, gpu=gpu)

        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(
                f"Time for grad_z iter:" f" {time_delta.total_seconds() * 1000}"
            )
        with torch.no_grad():
            tmp_influence = (
                -sum(
                    [
                        torch.sum(k * j).data
                        for k, j in zip(grad_z_vec, s_test_vec)
                    ]
                )
                / train_dataset_size
            )
        tmp_influence_np = tmp_influence.cpu().item()
        influences.append(-2.0 * tmp_influence_np)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num