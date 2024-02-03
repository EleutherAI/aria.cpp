import argparse
import sys
import struct
import torch
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(usage="convert-torch-to-ggml <flags> <state-dict-binary-file> <target>")
    parser.add_argument(
        "-size",
        help="size of model (medium, large).",
        choices=["medium", "large"],
        default="large"
    )
    parser.add_argument(
        "-fp32",
        help="use fp32 instead of fp16.",
        action="store_true"
    )
    parser.add_argument(
        "model_path",
        help="the file path of pytorch state dict binary file."
    )
    parser.add_argument(
        "target_path",
        help="the path of the target file.",
        default="aria-ggml.bin"
    )
    args = parser.parse_args(sys.argv[1:])

    with open(f"config/{args.size}.json", "r", encoding="utf-8") as f:
        hparams = json.load(f)
    print(hparams)
    list_vars = torch.load(args.model_path, map_location="cpu")
    print(list_vars.keys())

    with open(args.target_path, "wb") as fout:
        ftype = 0 if args.fp32 else 1
        fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
        fout.write(struct.pack("i", hparams["vocab_size"]))
        fout.write(struct.pack("i", hparams["max_seq_len"]))
        fout.write(struct.pack("i", hparams["d_model"]))
        fout.write(struct.pack("i", hparams["n_heads"]))
        fout.write(struct.pack("i", hparams["n_layers"]))
        fout.write(struct.pack("i", hparams["ff_mult"]))
        fout.write(struct.pack("i", ftype))

        """
        for i in range(hparams["vocab_size"]):
            text = tokenizer.decode([i]).encode('utf-8')
            fout.write(struct.pack("i", len(text)))
            fout.write(text)
        """


        for name in list_vars.keys():
            data = list_vars[name].squeeze().numpy()
            print("Processing variable: " + name + " with shape: ", data.shape)

            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                print("  Skipping variable: " + name)
                continue

            n_dims = len(data.shape)

            # ftype == 0 -> float32, ftype == 1 -> float16
            ftype_cur = 0
            if ftype != 0:
                if name.split(".")[-1] == "weight" and n_dims == 2:
                    print("  Converting to float16")
                    data = data.astype(np.float16)
                    ftype_cur = 1
                else:
                    print("  Converting to float32")
                    data = data.astype(np.float32)
                    ftype_cur = 0
            else:
                if data.dtype != np.float32:
                    print("  Converting to float32")
                    data = data.astype(np.float32)
                    ftype_cur = 0

            # header
            header = name.encode('utf-8')
            fout.write(struct.pack("iii", n_dims, len(header), ftype_cur))
            for i in range(n_dims):
                fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
            fout.write(header)

            # data
            data.tofile(fout)

    print("Done. Output file: " + args.target_path)
    print("")


if __name__ == "__main__":
    main()



