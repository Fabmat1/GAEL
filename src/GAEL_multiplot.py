import sys, math
import numpy as np

# import matplotlib
# matplotlib.use("module://mplcairo.qt")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

# ----------------------------------------------------------------------
def read_csv(path):
    """Return 1-D numpy arrays lam, flux, sigma, model, spline, ignoreflag"""
    data = np.genfromtxt(path, delimiter=',', names=True)
    return (data['lambda'],
            data['flux'],
            data['sigma'],
            data['model'],
            data['spline'],
            data['ignore'])

# ----------------------------------------------------------------------
def calculate_ylim_with_margin(values, margin=0.05):
    """Return y-limits with a relative margin, ignoring NaNs"""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1, 1
    ymin, ymax = finite.min(), finite.max()
    if ymin == ymax:                       # flat – make up a tiny range
        delta = 0.1 if ymin == 0 else 0.1 * abs(ymin)
        ymin, ymax = ymin - delta, ymax + delta
    else:
        rng = ymax - ymin
        ymin -= margin * rng
        ymax += margin * rng
    return ymin, ymax

# ----------------------------------------------------------------------
def split_masked(lam, values, ignore_mask):
    """
    Split an array into *good* and *bad* (ignored) parts – neighbouring
    elements next to a bad one are also masked to avoid spurious joins.
    """
    ok  = ignore_mask == 1
    bad = ~ok

    if bad.size > 1:
        transitions = bad[:-1] ^ bad[1:]
        guard = np.zeros_like(bad, dtype=bool)
        guard[:-1] |= transitions
        guard[1:]  |= transitions
        bad |= guard

    good_line = (lam, np.where(ok,  values, np.nan))
    bad_line  = (lam, np.where(bad, values, np.nan))
    return good_line, bad_line, ok                # also return boolean mask

# ----------------------------------------------------------------------
def enable_x_minor_ticks(ax, top=True, bottom=True):
    """Turn on minor ticks on the x axis and draw them on *top* / *bottom*"""
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='x', which='minor', direction='in',
                   top=top, bottom=bottom)

# ----------------------------------------------------------------------
def draw_segment(gs, row0, lam, flux, sig, mdl, spl, ign, x0, xrange):
    """Draw one 4-panel wavelength segment at grid row *row0*"""
    xlim = (x0, x0 + xrange)

    # ------------------------------------------------------------------
    # panel 1 – continuum (spline) + data
    ax1 = plt.subplot(gs[row0])
    (l_ok,f_ok),(l_bad,f_bad), good_mask = split_masked(lam, flux, ign)
    ax1.plot(l_ok,  f_ok,  'k',  lw=0.8)
    ax1.plot(l_bad, f_bad, color='0.6', ls=':', lw=0.8)
    ax1.plot(lam,  spl,  'r',  lw=1.5, zorder=10)
    ax1.set_xlim(xlim)
    ax1.set_ylabel('Continuum Flux')

    # y-limits: ignore *bad* points
    ymin, ymax = calculate_ylim_with_margin(
        np.concatenate([flux[good_mask], spl[np.isfinite(spl)]]))
    ax1.set_ylim(ymin, ymax)

    ax1.tick_params(axis='x', which='major', direction='in',
                    top=True, bottom=True, labelbottom=False)
    ax1.tick_params(axis='y', which='both', direction='in',
                    left=True, right=True)
    enable_x_minor_ticks(ax1, top=True, bottom=True)

    # ------------------------------------------------------------------
    # panel 2 – model + data (not normalised)
    ax2 = plt.subplot(gs[row0+1], sharex=ax1)
    (lm_ok,mm_ok),(lm_bad,mm_bad), good_m = split_masked(lam, mdl, ign)
    ax2.plot(lm_ok, mm_ok, 'r', lw=1.5, zorder=10)
    ax2.plot(lm_bad, mm_bad, color='r', ls=':', lw=1.5, zorder=10)
    ax2.plot(l_ok , f_ok , 'k',  lw=0.8)
    ax2.plot(l_bad, f_bad, color='0.6', ls=':', lw=0.8)
    ax2.set_xlim(xlim)
    ax2.set_ylabel('Full Flux')

    ymin, ymax = calculate_ylim_with_margin(
        np.concatenate([flux[good_mask], mdl[good_m]]))
    ax2.set_ylim(ymin, ymax)

    ax2.tick_params(axis='x', which='major', direction='in',
                    top=True, bottom=True, labelbottom=False)
    ax2.tick_params(axis='y', which='both', direction='in',
                    left=True, right=True)
    enable_x_minor_ticks(ax2, top=True, bottom=True)

    # ------------------------------------------------------------------
    # panel 3 – normalised
    ax3 = plt.subplot(gs[row0+2], sharex=ax1)
    nflux = flux / spl
    nmdl  = mdl  / spl
    (n_ok ,nf_ok),(n_bad ,nf_bad), good_nf = split_masked(lam, nflux, ign)
    (nm_ok,mf_ok),(nm_bad,mf_bad), good_nm = split_masked(lam, nmdl,  ign)
    ax3.plot(nm_ok, mf_ok, 'r', lw=1.5, zorder=10)
    ax3.plot(nm_bad,mf_bad, color='r', ls=':', lw=1.5, zorder=10)
    ax3.plot(n_ok , nf_ok, 'k',  lw=0.8)
    ax3.plot(n_bad,nf_bad, color='0.6', ls=':', lw=0.8)
    ax3.set_xlim(xlim)
    ax3.set_ylabel('Normalized Flux')

    ymin, ymax = calculate_ylim_with_margin(
        np.concatenate([nflux[good_nf], nmdl[good_nm]]))
    ax3.set_ylim(ymin, ymax)

    ax3.tick_params(axis='x', which='major', direction='in',
                    top=True, bottom=True, labelbottom=False)
    ax3.tick_params(axis='y', which='both', direction='in',
                    left=True, right=True)
    enable_x_minor_ticks(ax3, top=True, bottom=True)

    # ------------------------------------------------------------------
    # panel 4 – χ residuals
    ax4 = plt.subplot(gs[row0+3], sharex=ax1)
    chi = (flux - mdl) / sig
    (c_ok,chi_ok),(c_bad,chi_bad), _ = split_masked(lam, chi, ign)

    for y, ls in [(0, '-'), (1, '--'), (-1, '--'), (2, ':'), (-2, ':')]:
        ax4.axhline(y, color='grey' if y else 'k', ls=ls, lw=0.3, zorder=1)

    ax4.plot(c_ok , chi_ok , 'k',  lw=0.8)
    ax4.plot(c_bad, chi_bad, color='0.6', ls=':', lw=0.8)
    ax4.set_xlim(xlim)
    ax4.set_ylim(-5, 5)
    ax4.set_ylabel(r'$\chi$')
    ax4.set_xlabel('Wavelength [Å]')

    ax4.tick_params(axis='x', which='major', direction='in',
                    top=False, bottom=True)
    enable_x_minor_ticks(ax4, top=False, bottom=True)

# ----------------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)

    csv, pdf, xrange = sys.argv[1], sys.argv[2], float(sys.argv[3])
    lam, flux, sig, mdl, spl, ign = read_csv(csv)

    # wavelength segmentation -------------------------------------------
    lo, hi = lam[0], lam[-1]
    nseg   = math.ceil((hi - lo) / xrange)

    # DIN-A4 landscape in inches
    a4_w = 297 / 25.4          # 11.69 in
    a4_h = 210 / 25.4          #  8.27 in

    # grid: 3 tall panels + χ + spacer per segment
    h_ratios = sum(([3, 3, 3, 1, 1] for _ in range(nseg)), [])
    h_ratios.pop()             # no spacer after final block

    # height reduced to two-thirds
    total_h = a4_h * nseg * (2/3)
    fig = plt.figure(figsize=(a4_w, total_h))
    gs  = GridSpec(len(h_ratios), 1,
                   height_ratios=h_ratios,
                   hspace=0.0)

    for s in range(nseg):
        x0 = lo + s * xrange
        m  = (lam >= x0) & (lam <= x0 + xrange)
        draw_segment(gs, s * 5,
                     lam[m], flux[m], sig[m],
                     mdl[m], spl[m], ign[m],
                     x0, xrange)

    plt.tight_layout()
    plt.savefig(pdf)
    print("Wrote", pdf)

# ----------------------------------------------------------------------
if __name__ == '__main__':
    main()