"""Write a tiny random-init U-Net checkpoint for testing loaders against.

The real baseline checkpoint is 119 MiB and still stuck in Drive (see the
README's "Getting the weights out of Colab"). That blocks Person 4 from
writing `adapters/unet_torch.py`, because there is nothing to point a loader
at. This unblocks that half: same key names, same structure, same bare
`state_dict` format as the real file -- just untrained and 0.5 MB.

**These are not trained weights.** Anything evaluated with them produces
noise. The safeguard is the width: the fixture is (4,8,16,32) while the real
baseline is (64,128,256,512), so code that hardcodes the real widths hits a
shape mismatch and fails loudly instead of quietly reporting garbage. Use
`unet_model.load_unet`, which infers the width, and the same call works for
both files.

Run:
    python make_fixture_checkpoint.py
    python make_fixture_checkpoint.py --features 8,16,32,64 --out somewhere.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unet_model import UNet, features_from_state_dict, load_unet  # noqa: E402

HERE = Path(__file__).resolve().parent

# Deliberately narrower than the real baseline -- see the module docstring.
FIXTURE_FEATURES = (4, 8, 16, 32)
FIXTURE_SEED = 42


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--features",
        type=lambda s: tuple(int(x) for x in s.split(",") if x.strip()),
        default=FIXTURE_FEATURES,
    )
    parser.add_argument("--seed", type=int, default=FIXTURE_SEED)
    parser.add_argument(
        "--out", type=Path, default=HERE.parent / "tests" / "fixtures" / "unet_fixture_random.pt"
    )
    args = parser.parse_args(argv)

    # Seeded, so regenerating gives a byte-identical file and the commit
    # doesn't churn every time someone re-runs this.
    torch.manual_seed(args.seed)
    model = UNet(in_channels=3, out_channels=2, features=args.features)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Save a bare state_dict, exactly like the real checkpoint -- a loader
    # written against this must work unchanged against the trained file.
    torch.save(model.state_dict(), args.out)

    # Prove it round-trips before claiming success: reload through the same
    # helper Person 4 will use, and push one batch through it.
    reloaded = load_unet(args.out)
    with torch.no_grad():
        out = reloaded(torch.randn(1, 3, 256, 256))

    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out}  ({size_mb:.2f} MB)")
    print(f"  features inferred on reload: {features_from_state_dict(torch.load(args.out, map_location='cpu'))}")
    print(f"  forward pass 1x3x256x256 -> {tuple(out.shape)}")
    print("  NOT trained weights -- for exercising loader code only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
