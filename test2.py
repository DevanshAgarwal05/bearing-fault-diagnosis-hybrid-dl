import glob, os
import scipy.io as sio

files = glob.glob(r"C:\Users\devan\OneDrive\Desktop\SEM V\Soft_Computing_Project\data\raw\*.mat")
print("Found .mat files:", files)

for f in files:
    mat = sio.loadmat(f, struct_as_record=False, squeeze_me=True)
    print(f, "contains keys:", list(mat.keys()))
