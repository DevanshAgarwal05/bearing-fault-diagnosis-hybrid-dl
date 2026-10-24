import glob, os
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "*.mat")))
print("Found .mat files:", files)

for f in files:
    mat = sio.loadmat(f, struct_as_record=False, squeeze_me=True)
    print(f, "contains keys:", list(mat.keys()))
