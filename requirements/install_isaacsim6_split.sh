#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_PATH")")"
MODEL_VENV="$REPO_PATH/.venv-model312"
SIM_VENV="$REPO_PATH/.venv-isaacsim6"
ISAACLAB_PATH="$REPO_PATH/third_party/IsaacLab"
ISAACLAB_PATCH="$REPO_PATH/requirements/patches/isaaclab3_local_asset_root.patch"
ASSETS_ROOT="$(dirname "$REPO_PATH")/IsaacAssets6.0"
if [ -d "$(dirname "$REPO_PATH")/IsaacAssets6.0-minimal" ]; then
    ASSETS_ROOT="$(dirname "$REPO_PATH")/IsaacAssets6.0-minimal"
fi
DOWNLOAD_ROOT="$REPO_PATH/.downloads/isaacsim6"
INSTALL_MODEL=1
INSTALL_SIM=1
INSTALL_ASSETS=1

ISAACLAB_REF="release/3.0.0-beta2"
ISAACLAB_COMMIT="72cb3826d2dc5a90e697aaac824ecc3cb9e2752b"
ISAACSIM_VERSION="6.0.1.0"
RAY_VERSION="2.52.1"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
HYDRA_VERSION="1.3.2"
OMEGACONF_VERSION="2.3.0"

usage() {
    cat <<EOF
Usage: bash requirements/install_isaacsim6_split.sh [options]

Options:
  --model-venv PATH    Model/OpenPI venv (default: .venv-model312)
  --sim-venv PATH      Isaac Sim 6 venv (default: .venv-isaacsim6)
  --isaaclab-path PATH Clean Isaac Lab checkout
  --assets-root PATH   Isaac Sim 6 complete asset-pack root
  --skip-model         Do not install the model-side venv
  --skip-sim           Do not install Isaac Sim/Isaac Lab
  --skip-assets        Do not download the complete Isaac Sim 6 assets
  -h, --help           Show this help

Every network command is executed with HTTP(S)/SOCKS proxy variables removed.
Mirrors and Git URL rewrites are rejected.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-venv) MODEL_VENV="$(readlink -m "$2")"; shift 2 ;;
        --sim-venv) SIM_VENV="$(readlink -m "$2")"; shift 2 ;;
        --isaaclab-path) ISAACLAB_PATH="$(readlink -m "$2")"; shift 2 ;;
        --assets-root) ASSETS_ROOT="$(readlink -m "$2")"; shift 2 ;;
        --skip-model) INSTALL_MODEL=0; shift ;;
        --skip-sim) INSTALL_SIM=0; shift ;;
        --skip-assets) INSTALL_ASSETS=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

direct() {
    env \
        -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
        -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
        "$@"
}

check_direct_download_policy() {
    local git_network_rewrites
    git_network_rewrites="$(git config --show-origin --get-regexp \
        '(^http\..*\.proxy$|^https\..*\.proxy$|^http\.proxy$|^https\.proxy$|^url\..*\.insteadof$)' || true)"
    if [ -n "$git_network_rewrites" ]; then
        echo "Refusing network access because Git proxy/URL rewrite settings are active:" >&2
        echo "$git_network_rewrites" >&2
        echo "Remove them explicitly, then rerun. This script will not edit global Git config." >&2
        exit 1
    fi

    echo "Direct-download policy: proxy variables will be removed for every network command."
    if env | grep -Eiq '^(http|https|all|no)_proxy='; then
        echo "The parent shell contains proxy variables; they will not be inherited by downloads."
    fi
}

ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is required. Install it from https://docs.astral.sh/uv/ without a proxy." >&2
        exit 1
    fi
}

isaaclab_patch_is_applied() {
    git -C "$ISAACLAB_PATH" apply --reverse --check "$ISAACLAB_PATCH" \
        >/dev/null 2>&1
}

apply_isaaclab_patch() {
    if [ ! -f "$ISAACLAB_PATCH" ]; then
        echo "Isaac Lab compatibility patch not found: $ISAACLAB_PATCH" >&2
        exit 1
    fi
    if isaaclab_patch_is_applied; then
        echo "Isaac Lab local asset-root patch is already applied."
    elif git -C "$ISAACLAB_PATH" apply --check "$ISAACLAB_PATCH"; then
        git -C "$ISAACLAB_PATH" apply "$ISAACLAB_PATCH"
        echo "Applied Isaac Lab local asset-root patch."
    else
        echo "Isaac Lab local asset-root patch cannot be applied cleanly." >&2
        exit 1
    fi
}

install_model_env() {
    echo "Installing model-side environment at $MODEL_VENV"
    if "$MODEL_VENV/bin/python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
        direct env UV_HTTP_TIMEOUT=600 uv pip install --no-config --python "$MODEL_VENV/bin/python" \
            'openpi @ git+https://github.com/RLinf/openpi'
        local py_mm
        py_mm="$("$MODEL_VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        cp -r "$MODEL_VENV/lib/python$py_mm/site-packages/openpi/models_pytorch/transformers_replace/"* \
            "$MODEL_VENV/lib/python$py_mm/site-packages/transformers/"
    else
        direct bash "$REPO_PATH/requirements/install.sh" embodied \
            --model openpi --env isaaclab --venv "$MODEL_VENV" \
            --skip-simulator --no-root
    fi
    direct uv pip install --no-config --python "$MODEL_VENV/bin/python" \
        "ray[default]==$RAY_VERSION" accelerate tensorboard pytest \
        "hydra-core==$HYDRA_VERSION" "omegaconf==$OMEGACONF_VERSION"
    cat >> "$MODEL_VENV/bin/activate" <<EOF
export REPO_PATH="$REPO_PATH"
export PYTHONPATH="$REPO_PATH:\${PYTHONPATH:-}"
export RLINF_MODEL_PYTHON="$MODEL_VENV/bin/python"
export RLINF_ISAACSIM_PYTHON="$SIM_VENV/bin/python"
export ISAACSIM_ASSET_ROOT="$ASSETS_ROOT/Assets/Isaac/6.0"
export RLINF_SCENARIO_ASSET_ROOT="$REPO_PATH/rlinf/assets_isaaclab"
EOF
}

install_sim_env() {
    local actual_commit dirty_state expected_dirty_state

    echo "Installing Isaac Sim $ISAACSIM_VERSION at $SIM_VENV"
    if [ -x "$SIM_VENV/bin/python" ]; then
        local sim_python_minor
        sim_python_minor="$($SIM_VENV/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [ "$sim_python_minor" != "3.12" ]; then
            echo "Existing simulator venv must use Python 3.12: $SIM_VENV" >&2
            exit 1
        fi
        echo "Reusing existing Python 3.12 simulator venv."
    else
        direct uv venv "$SIM_VENV" --python 3.12 --seed
    fi
    # shellcheck disable=SC1091
    source "$SIM_VENV/bin/activate"
    direct uv pip install --no-config --upgrade pip
    direct env UV_HTTP_TIMEOUT=600 uv pip install --no-config \
        "isaacsim[all,extscache]==$ISAACSIM_VERSION" \
        --extra-index-url https://pypi.nvidia.com \
        --index-strategy unsafe-best-match --prerelease=allow
    direct env UV_HTTP_TIMEOUT=600 uv pip install --no-config --upgrade \
        "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url https://download.pytorch.org/whl/cu128

    if [ -d "$ISAACLAB_PATH/.git" ]; then
        actual_commit="$(git -C "$ISAACLAB_PATH" rev-parse HEAD)"
        dirty_state="$(git -C "$ISAACLAB_PATH" status --porcelain)"
        expected_dirty_state=" M source/isaaclab/isaaclab/utils/assets.py"
        if [ -n "$dirty_state" ]; then
            if [ "$actual_commit" != "$ISAACLAB_COMMIT" ] \
                || [ "$dirty_state" != "$expected_dirty_state" ] \
                || ! isaaclab_patch_is_applied; then
                echo "Isaac Lab checkout contains unexpected changes: $ISAACLAB_PATH" >&2
                echo "$dirty_state" >&2
                exit 1
            fi
            echo "Reusing Isaac Lab checkout with the expected local asset-root patch."
        elif [ "$actual_commit" != "$ISAACLAB_COMMIT" ]; then
            direct git -C "$ISAACLAB_PATH" fetch origin "$ISAACLAB_REF"
            git -C "$ISAACLAB_PATH" checkout --detach "$ISAACLAB_COMMIT"
        fi
    elif [ -e "$ISAACLAB_PATH" ]; then
        echo "Isaac Lab destination exists but is not a Git checkout: $ISAACLAB_PATH" >&2
        exit 1
    else
        mkdir -p "$(dirname "$ISAACLAB_PATH")"
        direct git clone --branch "$ISAACLAB_REF" --depth 1 \
            https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
    fi

    actual_commit="$(git -C "$ISAACLAB_PATH" rev-parse HEAD)"
    if [ "$actual_commit" != "$ISAACLAB_COMMIT" ]; then
        echo "Unexpected Isaac Lab commit: $actual_commit" >&2
        exit 1
    fi
    apply_isaaclab_patch
    direct "$ISAACLAB_PATH/isaaclab.sh" --install
    direct uv pip install --no-config "ray[default]==$RAY_VERSION" \
        "hydra-core==$HYDRA_VERSION" "omegaconf==$OMEGACONF_VERSION" \
        "imageio[ffmpeg]" gymnasium

    cat >> "$SIM_VENV/bin/activate" <<EOF
export REPO_PATH="$REPO_PATH"
export PYTHONPATH="$REPO_PATH:\${PYTHONPATH:-}"
export RLINF_MODEL_PYTHON="$MODEL_VENV/bin/python"
export RLINF_ISAACSIM_PYTHON="$SIM_VENV/bin/python"
export ISAAC_LAB_PATH="$ISAACLAB_PATH"
export ISAACSIM_ASSET_ROOT="$ASSETS_ROOT/Assets/Isaac/6.0"
export RLINF_SCENARIO_ASSET_ROOT="$REPO_PATH/rlinf/assets_isaaclab"
export OMNI_KIT_ACCEPT_EULA=YES
export VK_DRIVER_FILES="/etc/vulkan/icd.d/nvidia_icd.json"
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
EOF
    deactivate
}

download_assets() {
    if ! command -v aria2c >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
        echo "aria2c or curl is required to download the asset packs." >&2
        exit 1
    fi
    command -v unzip >/dev/null 2>&1 || {
        echo "unzip is required to extract the asset packs." >&2
        exit 1
    }
    mkdir -p "$DOWNLOAD_ROOT" "$ASSETS_ROOT"
    local checksums=(
        92149a1f50a21c0f04cca6507ab00653
        9b4b924e2d31bce41712d7637a0d6e42
        b1c62924beda91251d3f5318ffec2b00
        6bd7aa4d9b6c4161c2302e4c9418ade7
        c4a17942014be6b50492ae860496fef7
    )
    local part file url
    for part in 1 2 3 4 5; do
        file="isaac-sim-assets-complete-6.0.1.00${part}.zip"
        url="https://downloads.isaacsim.nvidia.com/$file"
        if command -v aria2c >/dev/null 2>&1; then
            direct aria2c -c --all-proxy="" --http-proxy="" --https-proxy="" \
                --checksum="md5=${checksums[$((part - 1))]}" \
                --dir="$DOWNLOAD_ROOT" "$url"
        else
            direct curl --fail --location --retry 5 --continue-at - \
                --output "$DOWNLOAD_ROOT/$file" "$url"
            echo "${checksums[$((part - 1))]}  $DOWNLOAD_ROOT/$file" | md5sum --check -
        fi
    done

    local combined="$DOWNLOAD_ROOT/isaac-sim-assets-complete-6.0.1.zip"
    if [ ! -f "$combined" ]; then
        direct bash -c 'cat "$1"/isaac-sim-assets-complete-6.0.1.00{1,2,3,4,5}.zip > "$2"' \
            bash "$DOWNLOAD_ROOT" "$combined"
    fi
    unzip -oq "$combined" -d "$ASSETS_ROOT"
    local asset_root="$ASSETS_ROOT/Assets/Isaac/6.0"
    if [ ! -d "$asset_root/Isaac" ] || [ ! -d "$asset_root/NVIDIA" ]; then
        echo "Incomplete Isaac Sim asset root: $asset_root" >&2
        exit 1
    fi
}

main() {
    check_direct_download_policy
    ensure_uv
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader || true
    [ "$INSTALL_MODEL" -eq 0 ] || install_model_env
    [ "$INSTALL_SIM" -eq 0 ] || install_sim_env
    [ "$INSTALL_ASSETS" -eq 0 ] || download_assets
    echo "Installation completed. Activate $MODEL_VENV to start Ray."
}

main
