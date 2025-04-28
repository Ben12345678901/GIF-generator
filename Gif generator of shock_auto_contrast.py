import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def auto_contrast(img, saturated=0.35):
    """
    Stretch img by clipping the lowest and highest `saturated` % of pixels.
    Returns (normalized_img, lo, hi), where lo/hi are your new display min/max.
    """
    lo, hi = np.percentile(img, (saturated, 100 - saturated))
    img_clipped = np.clip(img, lo, hi)
    return (img_clipped - lo) / (hi - lo), lo, hi

def create_comparison_gif(
    output_dir,
    gif_path,
    fps=1,
    cmaps=("inferno", "gray"),
    pixel_size=0.35,
    fig_title="Delay Comparison",
    saturated=0.35
):
    # 1) find & sort files by delay
    pattern = os.path.join(output_dir, "*_delay*_bgsub.npy")
    files   = glob.glob(pattern)
    rx      = re.compile(r"_delay(?P<d>[-+]?\d*\.?\d+)_bgsub\.npy$")
    entries = []
    for f in files:
        m = rx.search(os.path.basename(f))
        if m:
            entries.append((float(m.group("d")), f))
    entries.sort(key=lambda x: x[0])
    if not entries:
        raise RuntimeError(f"No matching .npy files in {output_dir}")

    # 2) pre-load arrays so we can compute a global auto‐contrast window
    arrays = [np.load(fp) for _, fp in entries]

    # shift all data so its minimum is zero
    vmin0 = min(a.min() for a in arrays)
    offset = -vmin0 if vmin0 < 0 else 0
    arrays = [a + offset for a in arrays]

    # compute the 0.35% / 99.65% global clip limits
    allpix = np.concatenate([a.ravel() for a in arrays])
    _, vmin_auto, vmax_auto = auto_contrast(allpix, saturated)

    # physical scaling
    h, w   = arrays[0].shape[:2]
    extent = [0, w * pixel_size, 0, h * pixel_size]

    # 3) set up figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    ims, cbars = [], []
    labels = ["Inferno", "Gray"]

    for ax, cmap, label in zip(axes, cmaps, labels):
        im = ax.imshow(
            arrays[0],
            cmap=cmap,
            origin="upper",
            extent=extent,
            vmin=vmin_auto,
            vmax=vmax_auto
        )
        ax.set_xlabel("µm")
        ax.set_ylabel("µm")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity")
        ims.append(im)
        cbars.append(cbar)

    # annotation on first panel
    txt = axes[0].text(
        0.02, 0.95, "",
        color="white",
        transform=axes[0].transAxes,
        fontsize=12,
        va="top",
        bbox=dict(facecolor='black', alpha=0.6, pad=2)
    )

    # 4) update each frame (same vmin/vmax all the way through)
    def update(frame_idx):
        delay, fp = entries[frame_idx]
        arr = np.load(fp) + offset
        for im in ims:
            im.set_data(arr)
        txt.set_text(f"Delay: {delay} ns")
        return ims + [txt]

    anim = FuncAnimation(fig, update, frames=len(entries), blit=True)

    # 5) save animation
    writer = PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    plt.close(fig)
    print(f"✔ Saved comparison GIF to {gif_path} "
          f"({len(entries)} frames at {fps} fps, contrast clipped to "
          f"[{vmin_auto:.1f}, {vmax_auto:.1f}])")


# Example usage
if __name__ == "__main__":
    output_dir = r"C:\Users\benny\OneDrive\Desktop\python\Physics toolbox\SACLA_Shock_Propagation\Analysis\Background_subtraction\CH_AU_Glue_Quartz"
    gif_path  = os.path.join(r"C:\Users\benny\OneDrive\Desktop\python\Physics toolbox\SACLA_Shock_Propagation\Analysis\shock propagation", "CH_AU_Glue_Quartz_contrast.gif")
    create_comparison_gif(
        output_dir,
        gif_path,
        fps=1,
        pixel_size=0.35,
        fig_title="Shock‐Propagation Delay Comparison",
        saturated=0.35
    )
