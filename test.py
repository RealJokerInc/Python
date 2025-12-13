import numpy as np
import matplotlib.pyplot as plt

def speed_ratio(s):
    """Return U_vertical / U_horizontal for s = l/a (array-like)."""
    s = np.asarray(s, dtype=float)
    L = np.log(s)
    return 2.0 * (L - 0.807) / (L + 0.193)

def main():
    # Range of slenderness ratios to visualize
    s = np.logspace(1.1, 6, 500)  # l/a from ~12.6 to 1e6
    R = speed_ratio(s)

    # Plot
    plt.figure(figsize=(7,5))
    plt.semilogx(s, R, linewidth=2)
    plt.axhline(2.0, linestyle='--', linewidth=1)  # the "2x slower" assumption
    plt.xlabel("Slenderness ratio  l/a")
    plt.ylabel("Speed ratio  U_vertical / U_horizontal")
    plt.title("Effect of Slenderness on Speed Ratio")
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plt.show()

    # Print a few reference values
    for sval in [20, 50, 100, 1e3, 1e4, 1e6]:
        print(f"l/a = {sval:>7g}  ->  Uv/Uh = {speed_ratio(sval):.3f}")

    # Estimate how slender you need to be to get within certain % of 2
    targets = [0.90, 0.95, 0.98]  # fractions of 2
    s_scan = np.logspace(1.1, 10, 20000)
    R_scan = speed_ratio(s_scan)
    for frac in targets:
        want = 2*frac
        idx = np.argmax(R_scan >= want)
        if R_scan[idx] >= want:
            print(f"Reach {frac*100:.0f}% of 2 (i.e., R≥{want:.2f}) at l/a ≈ {s_scan[idx]:.0f}")
        else:
            print(f"Could not reach {frac*100:.0f}% of 2 within scanned range.")

if __name__ == "__main__":
    main()

