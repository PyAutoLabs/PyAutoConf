import logging
import os
from pathlib import Path
os.environ['XLA_FLAGS'] = "--xla_disable_hlo_passes=constant_folding"

logger = logging.getLogger(__name__)

def check_jax_using_gpu(log_gpu_warning : bool = True) -> bool:

    import jax

    devices = jax.devices()

    # Normalize device_kind for matching
    for d in devices:
        kind = str(d).lower()

        no_gpu = False

        if "gpu" in kind or "cuda" in kind:
            print(f"GPU / CUDA detected: {d}")
        elif "tpu" in kind:
            print(f"TPU detected: {d}")
        elif "cpu" in kind:
            print(f"CPU detected: {d}")
            no_gpu = True
        else:
            print(f"Other device detected: {d}")
            no_gpu = True

    if no_gpu and log_gpu_warning:

        logger.info(
            """
            JAX did not detect a GPU or TPU device and is using the CPU for computations.

            PyAutoLens runs > 50 times faster with a GPU, so it is recommended to reinstall
            JAX With GPU support, if you have a GPU available, by following the JAX 
            installation instructions.
            """)

    return no_gpu

def _colab_setup(
    *,
    project_name: str,
    workspace_repo: str,
    workspace_dir: str,
    packages: list[str],
    raise_error_if_not_gpu: bool,
    gpu_error_message: str,
) -> None:
    """
    Shared Google Colab setup for the PyAuto* ecosystem.

    Assumes a `check_jax_using_gpu(log_gpu_warning: bool)` utility exists in scope.
    """
    import os
    import subprocess
    import sys

    # Identical JAX performance tweak.
    os.environ["XLA_FLAGS"] = "--xla_disable_hlo_passes=constant_folding"

    # --- Ensure we're in Colab and (optionally) have a GPU ---
    try:
        import google.colab  # noqa: F401

        no_gpu = check_jax_using_gpu(log_gpu_warning=False)

        if no_gpu and raise_error_if_not_gpu:
            raise RuntimeError(gpu_error_message)

    except ImportError:
        print(
            f"""
            You are not running in a Google Colab environment so cannot use the setup_colab() function.

            You should therefore have {project_name} installed locally in your environment already (e.g. via pip or
            conda) and can run the rest of your script normally.

            You may now continue running your script or Notebook.
            """
        )
        return

    # --- Install packages (no deps, consistent with your current approach) ---
    print()
    print(f"Now Installing {project_name} and setting up Colab Environment:")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *packages, "--no-deps"]
    )

    # --- Clone workspace ---
    try:
        subprocess.run(["git", "clone", workspace_repo, workspace_dir], check=True)
    except subprocess.CalledProcessError:
        print("Workspace already exists so not cloning again.")

    # --- Configure autoconf paths ---
    from autoconf import conf

    os.chdir(workspace_dir)

    conf.instance.push(
        new_path=Path(workspace_dir) / "config",
        output_path=Path(workspace_dir) / "output",
    )

    print(
        f"""
        ***Google Colab Setup Complete, which included:***

        - Installation of {project_name} and other required packages.
        - Cloning of the workspace GitHub repository.
        - Setting up environment variables for JAX for improved performance.
        - Setting the configuration paths to the workspace config and output folders suitable for Colab.
        """
    )


def for_autogalaxy(raise_error_if_not_gpu: bool = True) -> None:
    """
    Google Colab helper for PyAutoGalaxy.

    Parallel to `for_autolens`, but installs PyAutoGalaxy only (no `autolens`) and clones the
    `autogalaxy_workspace`.
    """
    packages = [
        # Core stack
        "autoconf",
        "autofit",
        "autoarray",
        "autogalaxy",
        # Notebook/runtime extras commonly used across workspaces
        "pyvis==0.3.2",
        "dill==0.4.0",
        "jaxnnls",
        "nautilus-sampler==1.0.4",
        "timeout_decorator==0.5.0",
        "anesthetic==2.8.14",
    ]

    gpu_error_message = """
    No GPU detected in Google Colab. PyAutoGalaxy runs much faster with a GPU, so switch GPU
    on in Colab settings.

    To do this:

    - Click "Runtime" in the top menu.
    - Click "Change runtime type".
    - Under "Hardware accelerator", select one of the "GPU" options available.

    You can set up Colab with a CPU (e.g. if GPUs are unavailable) by calling:

        setup_colab.for_autogalaxy(raise_error_if_not_gpu=False)
    """

    _colab_setup(
        project_name="PyAutoGalaxy",
        workspace_repo="https://github.com/Jammy2211/autogalaxy_workspace",
        workspace_dir="/content/autogalaxy_workspace",
        packages=packages,
        raise_error_if_not_gpu=raise_error_if_not_gpu,
        gpu_error_message=gpu_error_message,
    )

def for_autolens(raise_error_if_not_gpu: bool = True) -> None:
    """
    Google Colab helper for PyAutoLens.

    Installs the full PyAutoLens stack and clones the `autolens_workspace`.
    """
    packages = [
        # Core stack
        "autoconf",
        "autofit",
        "autoarray",
        "autogalaxy",
        "autolens",
        # Notebook/runtime extras commonly used across workspaces
        "pyvis==0.3.2",
        "dill==0.4.0",
        "jaxnnls",
        "nautilus-sampler==1.0.4",
        "timeout_decorator==0.5.0",
        "anesthetic==2.8.14",
    ]

    gpu_error_message = """
    No GPU detected in Google Colab. PyAutoLens runs > 50 times faster with a GPU, so switch GPU
    on in Colab settings.

    To do this:

    - Click "Runtime" in the top menu.
    - Click "Change runtime type".
    - Under "Hardware accelerator", select one of the "GPU" options available.

    You can set up Colab with a CPU (e.g. if GPUs are unavailable) by calling:

        setup_colab.for_autolens(raise_error_if_not_gpu=False)
    """

    _colab_setup(
        project_name="PyAutoLens",
        workspace_repo="https://github.com/Jammy2211/autolens_workspace",
        workspace_dir="/content/autolens_workspace",
        packages=packages,
        raise_error_if_not_gpu=raise_error_if_not_gpu,
        gpu_error_message=gpu_error_message,
    )
