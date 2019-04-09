from __future__ import absolute_import, division, print_function, unicode_literals
import errno
import hashlib
import os
import re
import shutil
import sys
import tempfile
import torch
import zipfile
from tools.shared.module_loader import import_module

if sys.version_info[0] == 2:
    from urlparse import urlparse
    from urllib2 import urlopen  # noqa f811
else:
    from urllib.request import urlopen
    from urllib.parse import urlparse  # noqa: F401

try:
    from tqdm import tqdm
except ImportError:
    # fake tqdm if it's not installed
    class tqdm(object):

        def __init__(self, total=None, disable=False):
            self.total = total
            self.disable = disable
            self.n = 0

        def update(self, n):
            if self.disable:
                return

            self.n += n
            if self.total is None:
                sys.stderr.write("\r{0:.1f} bytes".format(self.n))
            else:
                sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
            sys.stderr.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.disable:
                return

            sys.stderr.write('\n')

# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

MASTER_BRANCH = 'master'
ENV_TORCH_HUB_DIR = 'TORCH_HUB_DIR'
DEFAULT_TORCH_HUB_DIR = '~/.torch/hub'
VAR_DEPENDENCY = 'dependencies'
MODULE_HUBCONF = 'hubconf.py'
READ_DATA_CHUNK = 8192
hub_dir = None


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _git_archive_link(repo_owner, repo_name, branch):
    return 'https://github.com/{}/{}/archive/{}.zip'.format(repo_owner, repo_name, branch)


def _download_archive_zip(url, filename):
    sys.stderr.write('Downloading: \"{}\" to {}\n'.format(url, filename))
    response = urlopen(url)
    with open(filename, 'wb') as f:
        while True:
            data = response.read(READ_DATA_CHUNK)
            if len(data) == 0:
                break
            f.write(data)


def _load_attr_from_module(module, func_name):
    # Check if callable is defined in the module
    if func_name not in dir(module):
        return None
    return getattr(module, func_name)


def _setup_hubdir():
    global hub_dir
    if hub_dir is None:
        hub_dir = os.getenv(ENV_TORCH_HUB_DIR, DEFAULT_TORCH_HUB_DIR)

    hub_dir = os.path.expanduser(hub_dir)

    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)


def _parse_repo_info(github):
    branch = MASTER_BRANCH
    if ':' in github:
        repo_info, branch = github.split(':')
    else:
        repo_info = github
    repo_owner, repo_name = repo_info.split('/')
    return repo_owner, repo_name, branch


def _get_cache_or_reload(github, force_reload):
    # Parse github repo information
    repo_owner, repo_name, branch = _parse_repo_info(github)

    repo_dir = os.path.join(hub_dir, repo_name + '_' + branch)

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    if use_cache:
        sys.stderr.write('Using cache found in {}\n'.format(repo_dir))
    else:
        cached_file = os.path.join(hub_dir, branch + '.zip')
        _remove_if_exists(cached_file)

        url = _git_archive_link(repo_owner, repo_name, branch)
        _download_archive_zip(url, cached_file)

        # Github uses '{repo_name}-{branch_name}' as folder name which is not importable
        # We need to manually rename it to '{repo_name}_{branch_name}'
        # Github also renames folder repo-v1.x.x to repo-1.x.x
        cached_zipfile = zipfile.ZipFile(cached_file)
        extraced_repo_name = cached_zipfile.infolist()[0].filename
        extracted_repo = os.path.join(hub_dir, extraced_repo_name)
        _remove_if_exists(extracted_repo)
        # Unzip the code and rename the base folder
        cached_zipfile.extractall(hub_dir)

        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        shutil.move(extracted_repo, repo_dir)  # rename the repo

    return repo_dir


def _load_hubconf_module(github, force_reload):
    # Setup hub_dir to save downloaded files
    _setup_hubdir()

    repo_dir = _get_cache_or_reload(github, force_reload)

    m = import_module(MODULE_HUBCONF, repo_dir + '/' + MODULE_HUBCONF)

    return m


def _check_module_exists(name):
    if sys.version_info >= (3, 4):
        import importlib.util
        return importlib.util.find_spec(name) is not None
    elif sys.version_info >= (3, 3):
        # Special case for python3.3
        import importlib.find_loader
        return importlib.find_loader(name) is not None
    else:
        # NB: imp doesn't handle hierarchical module names (names contains dots).
        try:
            import imp
            imp.find_module(name)
        except Exception:
            return False
        return True


def _check_dependencies(m):
    dependencies = _load_attr_from_module(m, VAR_DEPENDENCY)

    if dependencies is not None:
        missing_deps = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        if len(missing_deps):
            raise RuntimeError('Missing dependencies: {}'.format(', '.join(missing_deps)))


def _load_entry_from_hubconf(m, model):
    if not isinstance(model, str):
        raise ValueError('Invalid input: model should be a string of function name')

    _check_dependencies(m)

    func = _load_attr_from_module(m, model)

    if func is None or not callable(func):
        raise RuntimeError('Cannot find callable {} in hubconf'.format(model))

    return func


def set_dir(d):
    r"""
    Optionally set hub_dir to a local dir to save downloaded models & weights.

    If this argument is not set, env variable `TORCH_HUB_DIR` will be searched first,
    `~/.torch/hub` will be created and used as fallback.

    Args:
        d: path to a local folder to save downloaded models & weights.
    """
    global hub_dir
    hub_dir = d


def list(github, force_reload=False):
    r"""
    List all entrypoints available in `github` hubconf.
    Args:
        github: Required, a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        force_reload: Optional, whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Returns:
        entrypoints: a list of available entrypoint names

    Example:
        >>> entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
    """

    hub_module = _load_hubconf_module(github, force_reload)

    # We take functions starts with '_' as internal helper functions
    entrypoints = [f for f in dir(hub_module) if callable(getattr(hub_module, f)) and not f.startswith('_')]

    return entrypoints


def show(github, model, force_reload=False):
    r"""
    Show the docstring of entrypoint `model`.
    Args:
        github: Required, a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model: Required, a string of entrypoint name defined in repo's hubconf.py
        force_reload: Optional, whether to discard the existing cache and force a fresh download.
            Default is `False`.
    Example:
        >>> torch.hub.show('pytorch/vision', 'resnet18', force_reload=True)
    """
    hub_module = _load_hubconf_module(github, force_reload)

    entry = _load_entry_from_hubconf(hub_module, model)

    print(entry.__doc__)


def load(github, model, force_reload=False, *args, **kwargs):
    r"""
    Load a model from a github repo, with pretrained weights.

    Args:
        github: Required, a string with format "repo_owner/repo_name[:tag_name]" with an optional
            tag/branch. The default branch is `master` if not specified.
            Example: 'pytorch/vision[:hub]'
        model: Required, a string of entrypoint name defined in repo's hubconf.py
        force_reload: Optional, whether to discard the existing cache and force a fresh download.
            Default is `False`.
        *args: Optional, the corresponding args for callable `model`.
        **kwargs: Optional, the corresponding kwargs for callable `model`.

    Returns:
        a single model with corresponding pretrained weights.

    Example:
        >>> model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    """
    hub_module = _load_hubconf_module(github, force_reload)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry(*args, **kwargs)


def _download_url_to_file(url, dst, hash_prefix, progress):
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.

    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/models`` where
    ``$TORCH_HOME`` defaults to ``~/.torch``. The default directory can be
    overridden with the ``$TORCH_MODEL_ZOO`` environment variable.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1)
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return torch.load(cached_file, map_location=map_location)
