import torch
import time
import datetime
import numpy as np
import copy
import logging
from tqdm import tqdm
from pathlib import Path

import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.autograd import grad
from torch.autograd.functional import vhp
from torch.utils.data import DataLoader
from scipy.optimize import fmin_ncg

# ===================== 新增：设备管理函数 =====================
def get_device(gpu_id):
    """获取设备（GPU/CPU），支持CUDA可用性检查"""
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    else:
        return torch.device("cpu")

# ===================== 原有辅助函数（无修改） =====================
def conjugate_gradient(ax_fn, b, debug_callback=None, avextol=None, maxiter=None):
    """Computes the solution to Ax - b = 0 by minimizing the conjugate objective
    f(x) = x^T A x / 2 - b^T x. This does not require evaluating the matrix A
    explicitly, only the matrix vector product Ax.

    From https://github.com/kohpangwei/group-influence-release/blob/master/influence/conjugate.py.

    Args:
      ax_fn: A function that return Ax given x.
      b: The vector b.
      debug_callback: An optional debugging function that reports the current optimization function. Takes two
          parameters: the current solution and a helper function that evaluates the quadratic and linear parts of the
          conjugate objective separately. (Default value = None)
      avextol:  (Default value = None)
      maxiter:  (Default value = None)

    Returns:
      The conjugate optimization solution.

    """

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


def make_functional(model):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
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
            set_attr(model, name.split("."), torch.nn.Parameter(p))


def tensor_to_tuple(vec, parameters):
    r"""Convert one vector to the parameters

    Adapted from
    https://pytorch.org/docs/master/generated/torch.nn.utils.vector_to_parameters.html#torch.nn.utils.vector_to_parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))

    # Pointer for slicing the vector for each parameter
    pointer = 0

    split_tensors = []
    for param in parameters:

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        split_tensors.append(vec[pointer:pointer + num_param].view_as(param))

        # Increment the pointer
        pointer += num_param

    return tuple(split_tensors)


def parameters_to_vector(parameters):
    r"""Convert parameters to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located

    vec = []
    for param in parameters:
        vec.append(param.view(-1))

    return torch.cat(vec)

# ===================== 核心函数（适配MNIST DataLoader） =====================
def s_test_cg(x_test, y_test, model, train_loader, damp, gpu=-1, verbose=True, loss_func="cross_entropy"):
    # 适配设备
    device = get_device(gpu)
    x_test, y_test = x_test.to(device), y_test.to(device)

    v_flat = parameters_to_vector(grad_z(x_test, y_test, model, gpu, loss_func=loss_func))

    def hvp_fn(x):

        x_tensor = torch.tensor(x, requires_grad=False).to(device)

        params, names = make_functional(model)
        # Make params regular Tensors instead of nn.Parameter
        params = tuple(p.detach().requires_grad_() for p in params)
        flat_params = parameters_to_vector(params)

        hvp = torch.zeros_like(flat_params).to(device)

        # 适配3元素DataLoader：添加train_ids
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


def s_test(x_test, y_test, model, i, samples_loader, gpu=-1, damp=0.01, scale=25.0, loss_func="cross_entropy"):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        x_test: torch tensor, test data points, such as test images
        y_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        i: the sample number
        samples_loader: torch DataLoader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor

    Returns:
        h_estimate: list of torch tensors, s_test"""
    # 统一模型状态
    model.eval()
    device = get_device(gpu)
    x_test, y_test = x_test.to(device), y_test.to(device)

    v = grad_z(x_test, y_test, model, gpu, loss_func=loss_func)
    h_estimate = v

    params, names = make_functional(model)
    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_().to(device) for p in params)

    # TODO: Dynamically set the recursion depth so that iterations stop once h_estimate stabilises
    progress_bar = tqdm(samples_loader, desc=f"IHVP sample {i}")
    # 适配3元素DataLoader：添加train_ids
    for i, (x_train, y_train, train_ids) in enumerate(progress_bar):
        x_train, y_train = x_train.to(device), y_train.to(device)

        def f(*new_params):
            load_weights(model, names, new_params)
            out = model(x_train)
            loss = calc_loss(out, y_train, loss_func=loss_func)
            return loss

        hv = vhp(f, params, tuple(h_estimate), strict=True)[1]

        # Recursively calculate h_estimate
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


def calc_loss(logits, labels, loss_func="cross_entropy"):
    """Calculates the loss

    Arguments:
        logits: torch tensor, input with size (minibatch, nr_of_classes)
        labels: torch tensor, target expected by loss of size (0 to nr_of_classes-1)
        loss_func: str, specify loss function name

    Returns:
        loss: scalar, the loss"""
    
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


def grad_z(x, y, model, gpu=-1, loss_func="cross_entropy"):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        x: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        y: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    device = get_device(gpu)
    x, y = x.to(device), y.to(device)

    prediction = model(x)
    loss = calc_loss(prediction, y, loss_func=loss_func)

    # Compute sum of gradients from model parameters to loss
    return grad(loss, model.parameters())


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
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        x_test: test image
        y_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor |
        scale: float, influence calculation scaling factor (to keep hessian <= I) | in the paper code use 25
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        loss_func: cross_entropy

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image
    """
    model.eval()
    device = get_device(gpu)
    x_test, y_test = x_test.to(device), y_test.to(device)

    """
    initialize inverse_hvp as a list of tensors with zeros, which should be first s_test as described in the paper
    H_0^(-1)v = v
    """
    inverse_hvp = [
        torch.zeros_like(params, dtype=torch.float).to(device) for params in model.parameters()
    ]

    for i in range(r): # repeat r times to get average
        hessian_loader = DataLoader(
            train_loader.dataset,
            sampler=torch.utils.data.RandomSampler(
                train_loader.dataset, True, num_samples=recursion_depth # as mentioned in paper, use "enought" samples
            ),
            batch_size=1,
            collate_fn=train_loader.collate_fn,  # 沿用MNIST的collate_fn
            # num_workers=4,
        )

        cur_estimate = s_test(
            x_test, y_test, model, i, hessian_loader, gpu=gpu, damp=damp, scale=scale, loss_func=loss_func,
        )

        with torch.no_grad():
            inverse_hvp = [
                old + (cur / scale) for old, cur in zip(inverse_hvp, cur_estimate) # update inverse_hvp by adding new cur_estimate
            ]

    with torch.no_grad():
        inverse_hvp = [component / r for component in inverse_hvp]

    return inverse_hvp


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
    """Calculates s_test for the whole test dataset taking into account all
    training data images.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        save: Path, path where to save the s_test files if desired. Omitting
            this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0

    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge.
        save: Path, path to the folder where the s_test files were saved to or
            False if they were not saved."""
    model.eval()
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    # 索引合法性校验
    if start < 0 or start >= len(test_loader.dataset):
        raise ValueError(f"start index {start} is out of range (0 to {len(test_loader.dataset)-1})")

    s_tests = []
    for i in range(start, len(test_loader.dataset)):
        # 适配MNIST Dataset：获取(x, y, idx)，忽略idx
        x_item, y_item, _ = test_loader.dataset[i]
        # 整理为collate_fn要求的输入格式（列表元素为(x,y,idx)）
        test_batch = [(x_item, y_item, i)]
        # 适配3元素collate_fn输出，提取x和y
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


def calc_grad_z(model, train_loader, save_pth=False, gpu=-1, start=0):
    """Calculates grad_z and can save the output to files. One grad_z should
    be computed for each training data sample.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        train_loader: pytorch dataloader, which can load the train data
        save_pth: Path, path where to save the grad_z files if desired.
            Omitting this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        start: int, index of the first test index to use. default is 0

    Returns:
        grad_zs: list of torch tensors, contains the grad_z tensors
        save_pth: Path, path where grad_z files were saved to or
            False if they were not saved."""
    model.eval()
    if save_pth and isinstance(save_pth, str):
        save_pth = Path(save_pth)
    if not save_pth:
        logging.info("ATTENTION: Not saving grad_z files!")

    # 索引合法性校验
    if start < 0 or start >= len(train_loader.dataset):
        raise ValueError(f"start index {start} is out of range (0 to {len(train_loader.dataset)-1})")

    grad_zs = []
    for i in range(start, len(train_loader.dataset)):
        # 适配MNIST Dataset：获取(x, y, idx)，忽略idx
        x_item, y_item, _ = train_loader.dataset[i]
        # 整理为collate_fn要求的输入格式
        train_batch = [(x_item, y_item, i)]
        # 适配3元素collate_fn输出，提取x和y
        z, t, _ = train_loader.collate_fn(train_batch)
        
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if save_pth:
            grad_z_vec = [g.cpu() for g in grad_z_vec]
            torch.save(grad_z_vec, save_pth.joinpath(f"{i}.grad_z"))
        else:
            grad_zs.append(grad_z_vec)

    return grad_zs, save_pth


def load_s_test(
    s_test_dir=Path("./s_test/"), s_test_id=0, r_sample_size=10, train_dataset_size=-1
):
    """Loads all s_test data required to calculate the influence function
    and returns a list of it.

    Arguments:
        s_test_dir: Path, folder containing files storing the s_test values
        s_test_id: int, number of the test data sample s_test was calculated
            for
        r_sample_size: int, number of s_tests precalculated
            per test dataset point
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        e_s_test: list of torch vectors, contains all e_s_tests for the whole
            dataset.
        s_test: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge."""
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
    ########################
    # TODO: should prob. not hardcode the file name, use natsort+glob
    ########################
    for i in range(num_s_test_files):
        try:
            s_test.append(torch.load(s_test_dir / str(s_test_id) + f"_{i}.s_test"))
        except FileNotFoundError:
            logging.warning(f"s_test file {s_test_id}_{i}.s_test not found, skipping...")
            continue

    #########################
    # TODO: figure out/change why here element 0 is chosen by default
    #########################
    e_s_test = s_test[0]
    # Calculate the sum
    for i in range(len(s_test)):
        e_s_test = [i + j for i, j in zip(e_s_test, s_test[0])]

    # Calculate the average
    #########################
    # TODO: figure out over what to calculate the average
    #       should either be r_sample_size OR e_s_test
    #########################
    e_s_test = [i / len(s_test) for i in e_s_test]

    return e_s_test, s_test


def load_grad_z(grad_z_dir=Path("./grad_z/"), train_dataset_size=-1):
    """Loads all grad_z data required to calculate the influence function and
    returns it.

    Arguments:
        grad_z_dir: Path, folder containing files storing the grad_z values
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        grad_z_vecs: list of torch tensors, contains the grad_z tensors"""
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


def calc_influence_function(train_dataset_size, grad_z_vecs=None, e_s_test=None):
    """Calculates the influence function

    Arguments:
        train_dataset_size: int, total train dataset size
        grad_z_vecs: list of torch tensor, containing the gradients
            from model parameters to loss
        e_s_test: list of torch tensor, contains s_test vectors

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness"""
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
                    ###################################
                    # TODO: verify if computation really needs to be done
                    # on the CPU or if GPU would work, too
                    ###################################
                    torch.sum(k * j).data.cpu().numpy()
                    for k, j in zip(grad_z_vecs[i], e_s_test)
                    ###################################
                    # Originally with [i] because each grad_z contained
                    # a list of tensors as long as e_s_test list
                    # There is one grad_z per training data sample
                    ###################################
                ]
            )
            / train_dataset_size
        )
        influences.append(tmp_influence)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()


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
    """Calculates the influences of all training data points on a single
    test dataset image.

    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size. | in the paper use 5000
        r: int, number of repeatation of which to take the avg. | in the paper use 10
            of the h_estimate calculation; r*recursion_depth should be less or equal to the
            training dataset size.
        gpu: int, identifies the gpu id, 0 for cpu
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    model.eval()
    device = get_device(gpu)

    # Calculate s_test vectors if not provided
    if s_test_vec is None:
        # 适配MNIST Dataset：获取(x, y, idx)，忽略idx
        x_item, y_item, _ = test_loader.dataset[test_id_num]
        # 整理为collate_fn要求的输入格式
        test_batch = [(x_item, y_item, test_id_num)]
        # 适配3元素collate_fn输出，提取x和y
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

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in tqdm(range(train_dataset_size)):
        # 适配MNIST Dataset：获取(x, y, idx)，忽略idx
        x_item, y_item, _ = train_loader.dataset[i]
        # 整理为collate_fn要求的输入格式
        train_batch = [(x_item, y_item, i)]
        # 适配3元素collate_fn输出，提取x和y
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
        influences.append(-tmp_influence_np)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num