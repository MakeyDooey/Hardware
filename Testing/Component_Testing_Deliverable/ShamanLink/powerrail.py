import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path

# Try compatible LTspice parsers
try:
    from ltspice import Ltspice
except ImportError:
    try:
        from PyLTSpice import RawRead as Ltspice
    except ImportError:
        raise ImportError("Install ltspice or PyLTSpice:\n   pip install ltspice\n")


# -----------------------------
# CONFIGURATION
# -----------------------------
RAW_FILE = "ShamanLink_PowerRail.raw"  # <-- replace with your .raw file

# Numeric test bounds (3.3V +/-5%)
NOMINAL = 3.3
LOWER_SPEC = 3.135
UPPER_SPEC = 3.465

# Test timing (match the LTSpice .meas definitions in your plan)
# Load step occurs at 10 ms
LOAD_STEP_TIME = 0.010  # seconds (10 ms)
# Recovery allowed (ms)
RECOVERY_LIMIT_MS = 10
# Pre-load average window (match .meas AVG FROM=2m TO=9m)
PRE_AVG_FROM = 0.002
PRE_AVG_TO = 0.009
# Ripple measurement window after settling
RIPPLE_START = 0.030     # 30 ms
RIPPLE_END = 0.050       # 50 ms
# -----------------------------


def load_raw(path):
    print(f"\nLoading LTspice RAW: {path}")
    raw = Ltspice(path)
    raw.parse()
    return raw


def extract_waveform(raw, node="V(3V3)"):
    """
    Extract time and waveform for `node` from different LTspice parser objects.

    Supports APIs from the `ltspice` package (has `variables`, `time`, `get_data` / `data`)
    and `PyLTSpice.RawRead` (has `get_trace`, `get_trace_names`, `get_time_axis`, `get_wave`).
    """
    # Case 1: ltspice (provides .variables / .time and get_data or .data)
    if hasattr(raw, "variables"):
        # try exact match, then case-insensitive match
        if node in raw.variables:
            idx = raw.variables.index(node)
        else:
            # case-insensitive lookup
            low = node.lower()
            found = None
            for i, name in enumerate(raw.variables):
                try:
                    if name.lower() == low:
                        found = i
                        break
                except Exception:
                    continue
            if found is None:
                print("Available variables:")
                print(raw.variables)
                raise ValueError(f"Node {node} not found in .raw file.")
            idx = found

        # Prefer full raw arrays if available (x_raw/y_raw include stepped cases)
        if hasattr(raw, 'x_raw') and hasattr(raw, 'y_raw'):
            time = np.array(raw.x_raw)
            # y_raw shape: (n_samples, n_variables)
            try:
                v = np.array(raw.y_raw)[:, idx]
            except Exception:
                # fallback to get_data if direct column extraction fails
                if hasattr(raw, "get_data"):
                    try:
                        v = np.array(raw.get_data(node))
                    except TypeError:
                        v = np.array(raw.get_data(idx))
                elif hasattr(raw, "data"):
                    v = np.array(raw.data[idx])
                else:
                    raise RuntimeError("Unable to retrieve trace data from LTspice raw object.")
        else:
            # time can be an attribute or available through get_time()/get_x
            if hasattr(raw, "time"):
                time = np.array(raw.time)
            elif hasattr(raw, "get_time"):
                time = np.array(raw.get_time())
            elif hasattr(raw, "get_x"):
                time = np.array(raw.get_x())
            else:
                raise RuntimeError("Unable to find time axis on LTspice raw object.")

            # waveform data: prefer get_data(name) (some versions expect a variable name)
            if hasattr(raw, "get_data"):
                try:
                    v = np.array(raw.get_data(node))
                except TypeError:
                    # fallback if implementation expects an index
                    v = np.array(raw.get_data(idx))
            elif hasattr(raw, "data"):
                v = np.array(raw.data[idx])
            else:
                raise RuntimeError("Unable to retrieve trace data from LTspice raw object.")

        return time, v

    # Case 2: PyLTSpice RawRead-style API
    if hasattr(raw, "get_trace_names") or hasattr(raw, "get_trace"):
        # get list of traces
        try:
            names = raw.get_trace_names()
        except Exception:
            # some versions use get_plot_names / get_plot_name(s)
            names = None
            for candidate in ("get_plot_names", "get_plot_name", "get_wave", "get_plot_name"):
                if hasattr(raw, candidate):
                    try:
                        names = getattr(raw, candidate)()
                        break
                    except Exception:
                        names = None
        if names is not None and node not in names:
            print("Available variables:")
            print(names)
            raise ValueError(f"Node {node} not found in .raw file.")

        # Prefer get_trace which often returns (time, values)
        if hasattr(raw, "get_trace"):
            t_v = raw.get_trace(node)
            # get_trace may return a tuple (time, values) or just values
            if isinstance(t_v, tuple) and len(t_v) >= 2:
                time = np.array(t_v[0])
                v = np.array(t_v[1])
                return time, v
            else:
                # fallback: get time axis and the wave
                if hasattr(raw, "get_time_axis"):
                    time = np.array(raw.get_time_axis())
                else:
                    raise RuntimeError("Unable to determine time axis for PyLTSpice raw object.")
                # try get_wave / get_trace again differently
                if hasattr(raw, "get_wave"):
                    v = np.array(raw.get_wave(node))
                else:
                    v = np.array(t_v)
                return time, v

        # Last-resort: try get_wave + get_time_axis
        if hasattr(raw, "get_wave") and hasattr(raw, "get_time_axis"):
            time = np.array(raw.get_time_axis())
            v = np.array(raw.get_wave(node))
            return time, v

    # If we get here we couldn't find a supported interface
    raise RuntimeError("Unsupported raw parser object: cannot extract waveform.\n"
                       "Object members: " + ",".join([m for m in dir(raw) if not m.startswith('_')]))


def find_recovery_time(time, v, step_t, lower_limit):
    """
    Finds the first time after the load step where V >= lower_limit.
    """
    mask = time >= step_t
    t2 = time[mask]
    v2 = v[mask]
    above = np.where(v2 >= lower_limit)[0]
    if len(above) == 0:
        return None
    return t2[above[0]]


def evaluate_results(vmin_post, vmax_post, overshoot, recovery_ms, ripple):
    results = {}
    # DC Range evaluated on post-startup waveform (ignore initial 0V at power-on)
    results["DC Range"] = "PASS" if (LOWER_SPEC <= vmin_post <= UPPER_SPEC and LOWER_SPEC <= vmax_post <= UPPER_SPEC) else "FAIL"
    results["Overshoot"] = "PASS" if overshoot <= 5 else "FAIL"
    results["Recovery Time"] = "PASS" if (recovery_ms is not None and recovery_ms <= RECOVERY_LIMIT_MS) else "FAIL"
    # Allow larger ripple tolerance per user request (0.2 V pk-pk)
    results["Ripple"] = "PASS" if ripple <= 0.2 else "FAIL"

    return results


def print_summary(case_label, vmin_overall, vmin_post, vmax_post, vavg, overshoot, recovery_ms, ripple, results):
    print(f"\n===================== POWER RAIL TEST RESULTS [{case_label}] =====================")
    print(f"Vmin (overall): {vmin_overall:.4f} V")
    print(f"Vmin (post-startup): {vmin_post:.4f} V (limit >= {LOWER_SPEC} V)")
    print(f"Vmax (post-startup): {vmax_post:.4f} V (limit <= {UPPER_SPEC} V)")
    print(f"Vavg:           {vavg:.4f} V (pre-load {PRE_AVG_FROM*1000:.0f}ms-{PRE_AVG_TO*1000:.0f}ms)")
    print(f"Overshoot:      {overshoot:.2f} % (limit <= 5%)")
    if recovery_ms is None:
        print("Recovery Time:  FAILED (rail never recovered above lower spec)")
    else:
        print(f"Recovery Time:  {recovery_ms:.3f} ms (limit <= {RECOVERY_LIMIT_MS} ms)")
    print(f"Ripple pk-pk:   {ripple*1000:.2f} mV (limit <= 200 mV)")
    print("\n------------------------ PASS / FAIL ------------------------------")
    for k, v in results.items():
        print(f"{k:15}: {v}")
    print("==================================================================\n")


def draw_test_guides(x_in_ms=True):
    """Draw test criteria guides on the current matplotlib axes.

    - Horizontal dashed red lines: LOWER_SPEC and UPPER_SPEC
    - Horizontal dotted green line: NOMINAL
    - Vertical dashed orange line: LOAD_STEP_TIME (in ms when x_in_ms=True)

    This function is intentionally small and uses plt directly so it can be
    called right before showing the figure.
    """
    if plt is None:
        return

    # horizontal spec lines
    try:
        # draw lines and give them labels; duplicates will be collapsed in legend
        plt.axhline(LOWER_SPEC, color="red", linestyle="--", linewidth=1.0, label=f"Lower spec {LOWER_SPEC} V")
        plt.axhline(UPPER_SPEC, color="red", linestyle="--", linewidth=1.0, label=f"Upper spec {UPPER_SPEC} V")
        plt.axhline(NOMINAL, color="green", linestyle=":", linewidth=1.0, label=f"Nominal {NOMINAL} V")

        # vertical: load step time (convert to ms if plots use ms on x-axis)
        x = LOAD_STEP_TIME * 1000.0 if x_in_ms else LOAD_STEP_TIME
        plt.axvline(x, color="orange", linestyle="-.", linewidth=1.0, label=f"Load step {LOAD_STEP_TIME*1000:.1f} ms")

        # make legend labels unique (avoid duplicate labels when guides are added multiple times)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        new_handles = []
        new_labels = []
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = True
                new_handles.append(h)
                new_labels.append(l)
        ax.legend(new_handles, new_labels)
    except Exception:
        # plotting should never crash the analyzer; ignore errors
        pass


def compute_plot_ylim(time, v, use_post_from=True, pad_frac=0.08, min_pad=0.01):
    """Compute a Y-axis limit focused on the post-startup region.

    - If use_post_from is True, use samples where time >= PRE_AVG_FROM.
    - Adds fractional padding (pad_frac) around the min/max. Ensures a
      small minimum padding of min_pad volts to avoid flat lines.
    """
    try:
        t = np.array(time)
        y = np.array(v)
        if use_post_from:
            mask = t >= PRE_AVG_FROM
            if np.any(mask):
                y2 = y[mask]
            else:
                y2 = y
        else:
            y2 = y

        ymin = float(np.min(y2))
        ymax = float(np.max(y2))
        span = max(ymax - ymin, 0.0)
        pad = max(span * pad_frac, min_pad)
        return (ymin - pad, ymax + pad)
    except Exception:
        # fallback: no limits
        return None


def analyze_waveform(time, v, case_label="case"):
    """Compute metrics for a single time-series and print summary."""
    # Pre-step region for DC average: use configured PRE_AVG_FROM..TO
    pre_mask = (time >= PRE_AVG_FROM) & (time <= PRE_AVG_TO)
    if np.any(pre_mask):
        v_pre = v[pre_mask]
        vavg = float(np.mean(v_pre))
    else:
        vavg = float(np.mean(v[time < LOAD_STEP_TIME]))

    vmin = float(np.min(v))
    vmax = float(np.max(v))

    # Evaluate post-startup (ignore initial power-on at 0V)
    post_mask = time >= PRE_AVG_FROM
    if np.any(post_mask):
        vmin_post = float(np.min(v[post_mask]))
        vmax_post = float(np.max(v[post_mask]))
    else:
        vmin_post = vmin
        vmax_post = vmax

    overshoot = 100.0 * (vmax_post - NOMINAL) / NOMINAL

    # Recovery time (first time after load step where V >= LOWER_SPEC)
    t_recov = find_recovery_time(time, v, LOAD_STEP_TIME, LOWER_SPEC)
    recovery_ms = None if t_recov is None else (t_recov - LOAD_STEP_TIME) * 1000.0

    # Ripple pk-pk in configured window
    ripple_mask = (time >= RIPPLE_START) & (time <= RIPPLE_END)
    used_ripple_window = None
    if np.any(ripple_mask):
        ripple_vals = v[ripple_mask]
        used_ripple_window = (RIPPLE_START, RIPPLE_END)
    else:
        # fallback: use last 10 ms of this case (if available), otherwise last 20% of case
        case_duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
        last_ms = 0.010
        if case_duration >= last_ms:
            start = time[-1] - last_ms
        else:
            start = time[0] + 0.8 * case_duration
        fallback_mask = (time >= start) & (time <= time[-1])
        if np.any(fallback_mask):
            ripple_vals = v[fallback_mask]
            used_ripple_window = (start, float(time[-1]))
        else:
            # last resort: entire case
            ripple_vals = v
            used_ripple_window = (float(time[0]), float(time[-1]))

    ripple = float(np.max(ripple_vals) - np.min(ripple_vals))

    results = evaluate_results(vmin_post, vmax_post, overshoot, recovery_ms, ripple)

    # attach ripple window info to results for transparency
    results_out = results.copy()
    results_out["ripple_window"] = used_ripple_window
    print_summary(case_label, vmin, vmin_post, vmax_post, vavg, overshoot, recovery_ms, ripple, results)

    return {
        "vmin": vmin,
        "vmax": vmax,
        "vavg": vavg,
        "overshoot_pct": overshoot,
        "recovery_ms": recovery_ms,
        "ripple_pkpk": ripple,
        "results": results,
        "ripple_window": used_ripple_window,
    }


def main():
    parser = argparse.ArgumentParser(description="Process LTSpice RAW files for 3.3V rail tests")
    parser.add_argument("rawfiles", nargs="*", help="One or more .raw files to analyze (defaults to RAW_FILE)")
    parser.add_argument("--node", default="V(3V3)", help="Node name to extract (default: V(3V3))")
    parser.add_argument("--out", help="Optional output JSON file to write aggregated results")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    files = args.rawfiles if args.rawfiles else [RAW_FILE]
    aggregated = []

    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"Raw file not found: {f}")
            continue

        raw = load_raw(str(path))
        try:
            time, v = extract_waveform(raw, node=args.node)
        except Exception as e:
            print(f"Error extracting node {args.node} from {f}: {e}")
            continue

        # If the raw parser provides explicit case split points (from .step)
        case_splits = getattr(raw, '_case_split_point', None)
        if case_splits and len(case_splits) >= 2:
            # case_splits are sample indices into the time axis
            for k in range(len(case_splits)-1):
                s = case_splits[k]
                e = case_splits[k+1]
                # slice time
                tseg = time[s:e]
                # slice voltage depending on shape
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    vseg = v[s:e]
                elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == len(time):
                    # time along first axis
                    vseg = v[s:e, :].mean(axis=1)
                elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[-1] == len(time):
                    # time along last axis
                    vseg = v[:, s:e].mean(axis=0)
                else:
                    # fallback to original v
                    vseg = v

                case_label = f"{path.stem}_case_{k}"
                metrics = analyze_waveform(tseg, vseg, case_label=case_label)
                aggregated.append({"file": str(path), "case": k, **metrics})
                if not args.no_plot:
                    plt.plot(tseg * 1000, vseg, label=case_label)

            if not args.no_plot:
                plt.xlabel("Time (ms)")
                plt.ylabel(f"{args.node} [V]")
                plt.title(f"3.3V Rail Transient Response ({path.stem} stepped cases)")
                plt.grid()
                # set fixed y-axis to 2V..4V for clarity
                plt.ylim(2.0, 4.0)
                # add test guides (spec lines / nominal / load-step)
                draw_test_guides(x_in_ms=True)
                plt.show()
            # continue to next file after processing splits
            continue

        # If v is 2D it's likely a stepped simulation (multiple cases). Handle per-case.
        if isinstance(v, np.ndarray) and v.ndim == 2:
            # Determine which axis corresponds to cases
            if hasattr(raw, "case_count") and raw.case_count and raw.case_count > 1:
                ncases = int(raw.case_count)
            else:
                # try to infer from shape
                if v.shape[0] > v.shape[1]:
                    ncases = v.shape[0]
                else:
                    ncases = v.shape[1]

            # determine case index ranges from raw parser if available
            case_splits = getattr(raw, '_case_split_point', None)
            case_ranges = []
            if case_splits and len(case_splits) >= 2:
                for k in range(len(case_splits)-1):
                    case_ranges.append((case_splits[k], case_splits[k+1]))
            else:
                # evenly split the time axis
                total = v.shape[-1]
                step = total // ncases
                for k in range(ncases):
                    start = k*step
                    end = (k+1)*step if k < ncases-1 else total
                    case_ranges.append((start, end))

            # try to find a selector node name (V(n001) or similar) in variables
            sel_var = None
            if hasattr(raw, 'variables'):
                cand_patterns = ['v(n001)', 'v(n002)', 'v(sel)', 'v(vsel)', 'sel']
                for pat in cand_patterns:
                    for name in raw.variables:
                        if pat in name.lower():
                            sel_var = name
                            break
                    if sel_var:
                        break
                # fallback: pick a variable (not time) that looks like a step (few distinct means)
                if not sel_var:
                    for name in raw.variables:
                        if name.lower() == 'time':
                            continue
                        try:
                            yfull = np.array(raw.get_data(name))
                        except Exception:
                            continue
                        means = []
                        for (s,e) in case_ranges:
                            if yfull.ndim == 2:
                                yslice = yfull[:, s:e].mean(axis=1)
                                m = float(np.mean(yslice))
                            else:
                                yslice = yfull[s:e]
                                m = float(np.mean(yslice))
                            means.append(round(m, 6))
                        if len(set(means)) > 1 and len(set(means)) <= max(2, ncases):
                            sel_var = name
                            break

            # iterate over cases
            for i in range(ncases):
                # pick correct slice
                if v.shape[0] == ncases:
                    vi = v[i, :]
                elif v.shape[1] == ncases:
                    vi = v[:, i]
                else:
                    # fallback: take first axis
                    vi = v[i % v.shape[0], :]

                # determine selector value for this case if available
                sel_val = None
                if sel_var:
                    try:
                        yfull = np.array(raw.get_data(sel_var))
                        s,e = case_ranges[i]
                        if yfull.ndim == 2:
                            yslice = yfull[:, s:e].mean(axis=1)
                            sel_val = float(np.mean(yslice))
                        else:
                            sel_val = float(np.mean(yfull[s:e]))
                    except Exception:
                        sel_val = None

                if sel_val is not None:
                    sval = int(round(sel_val)) if abs(sel_val - round(sel_val)) < 1e-6 else round(sel_val, 6)
                    case_label = f"{path.stem}_SEL={sval}_case_{i}"
                else:
                    case_label = f"{path.stem}_case_{i}"

                metrics = analyze_waveform(time, vi, case_label=case_label)
                aggregated.append({"file": str(path), "case": i, **metrics})
                # Optional: plot each case
                if not args.no_plot:
                    plt.plot(time * 1000, vi, label=case_label)

            plt.xlabel("Time (ms)")
            plt.ylabel("V(3V3) [V]")
            plt.title("3.3V Rail Transient Response (stepped cases)")
            plt.grid()
            if not args.no_plot:
                # set fixed y-axis to 2V..4V for clarity
                plt.ylim(2.0, 4.0)
                draw_test_guides(x_in_ms=True)
                plt.show()
            return

        # Single-case waveform
        metrics = analyze_waveform(time, v, case_label=path.stem)
        aggregated.append({"file": str(path), "case": 0, **metrics})

        # Optional plot
        if not args.no_plot:
            plt.plot(time * 1000, v)
            plt.xlabel("Time (ms)")
            plt.ylabel("V(3V3) [V]")
            plt.title("3.3V Rail Transient Response")
            plt.grid()
            # set fixed y-axis to 2V..4V for clarity
            plt.ylim(2.0, 4.0)
            draw_test_guides(x_in_ms=True)
            plt.show()

    # Write aggregated results if requested
    if args.out:
        outp = Path(args.out)
        with outp.open("w", encoding="utf-8") as fh:
            json.dump(aggregated, fh, indent=2)
        print(f"Wrote aggregated results to {outp}")


if __name__ == "__main__":
    main()
