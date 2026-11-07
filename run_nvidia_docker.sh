# ONLY IF TENSORFLOW DOES NOT RECOGNIZE THE GPU. RUN with: source ./run_nvidia_docker.sh
NVIDIA_LIB_PATHS=$(python3 - <<'PY'
import site
import glob
import os

paths = []
for root in site.getsitepackages() + [site.getusersitepackages()]:
    paths.extend(glob.glob(os.path.join(root, "nvidia", "*", "lib")))

print(":".join(p for p in paths if os.path.isdir(p)))
PY
)

export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"