import torch
from tqdm import tqdm
import argparse
import os
from MoA.attention.convert import layout_to_lut_single_density

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_path", type=str, required=True, help="path to load the layout")
    parser.add_argument("--lut_path", type=str, default=None, help="path to save the lut")

    args = parser.parse_args()

    # default lut path is the same folder as layout path
    if(args.lut_path is None):
        # get the filename
        filename = os.path.basename(args.layout_path)

        if "layout" in filename:
            filename = filename.replace("layout", "lut")
            args.lut_path = os.path.join(os.path.dirname(args.layout_path), filename)
        else:
            args.lut_path = args.layout_path.replace(".pt", "_lut.pt")

    layout = torch.load(args.layout_path)

    lut = layout_to_lut_single_density(layout)
    torch.save(lut, args.lut_path)

# l=[]
# for i in range(32):
#     n = i+1
#     N = 32
#     # print('density:', (n*(n+1)/2+n*(N-n))/(N*(N+1)/2))
#     l.append((n*(n+1)/2+n*(N-n))/(N*(N+1)/2))
# print(l)