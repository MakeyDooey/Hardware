import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import datetime
import csv
import binascii
import os
import atexit
import threading
import queue

# -------------------------------
# Configuration
# -------------------------------
# Replace with your Nucleo's virtual COM port or set to None to try auto-detect
COM_PORT = "COM7"
BAUD_RATE = 115200
MAX_POINTS = 100

# Quick mitigation / visualization options
# If True, values that look like Fahrenheit (>F_THRESHOLD) will be converted to Celsius
AUTO_CONVERT_F_TO_C = False
F_TO_C_THRESHOLD = 80.0  # If parsed value > this, assume Fahrenheit

# Exponential moving average for plotting (don't modify raw logged values)
USE_EMA = True
EMA_ALPHA = 0.2  # smoothing factor for EMA: higher = less smoothing

# Temporary override: if parsed temp looks obviously wrong (very large),
# divide it by DIVIDE_FACTOR before masking/EMA/plotting. This preserves
# `orig_temp` in the CSV while making the live plot reasonable.
ENABLE_DIVIDE_OVERRIDE = True
DIVIDE_THRESHOLD = 120.0
DIVIDE_FACTOR = 8.0

# Quick masking options: if the parsed temperature is clearly out-of-range
# (e.g. due to firmware/ADC issues), mask the plotted value by using the
# EMA/previous value or a clamp so the graph remains readable. Raw values
# and diagnostics are still logged to CSV and debug log.
ENABLE_MASKING = True
MIN_DISPLAY_TEMP = -40.0
MAX_DISPLAY_TEMP = 85.0

# Logging
LOG_FILE = "temperature_log.csv"

# Assumptions about incoming data:
# - Legacy single float per line: "23.5\n"
# - Structured CSV: "seq,temp,error_flag,device_timestamp,crc_hex"
#   Example: "123,24.7,0,1699070000.123,1a2b3c4d"
#   CRC is CRC32 (hex) of the payload before the crc field, encoded as utf-8.
# The parser will try the structured form first and gracefully fall back to legacy.


def find_mcu_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'STM32' in port.description or 'USB Serial' in port.description or 'Nucleo' in port.description:
            return port.device
    return None


# -------------------------------
# Initialize Serial
# -------------------------------
if COM_PORT is None:
    detected = find_mcu_port()
    if detected:
        COM_PORT = detected
    else:
        raise RuntimeError("No COM port specified and auto-detect failed")

ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)


# -------------------------------
# Prepare real-time plot
# -------------------------------
try:
    plt.style.use('seaborn-darkgrid')
except OSError:
    # If seaborn is not installed or the style is not available, fall back
    # to the default matplotlib style and warn the user. Installing seaborn
    # (pip install seaborn) will provide the requested style.
    print("Warning: 'seaborn-darkgrid' style not found; falling back to default style."
          " To use seaborn styles install the 'seaborn' package.")
    plt.style.use('default')
fig, ax = plt.subplots()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('STM32 Temperature Readings')

x_data, y_data = [], []
line, = ax.plot([], [], lw=2)
start_time = None
ema = None
last_good_display = None
line_queue = queue.Queue()


def serial_reader(ser, q):
    # Continuously read lines from serial and push (recv_time, bytes) to queue
    while True:
        try:
            b = ser.readline()
            if b:
                q.put((time.time(), b))
        except Exception:
            # On serial error, stop reader
            break


# start background reader thread so plotting and serial I/O are decoupled
reader_thread = threading.Thread(target=serial_reader, args=(ser, line_queue), daemon=True)
reader_thread.start()


# -------------------------------
# CSV Logger setup
# -------------------------------
log_file = open(LOG_FILE, 'a', newline='')
csv_writer = csv.writer(log_file)
if os.stat(LOG_FILE).st_size == 0:
    csv_writer.writerow([
        'host_iso_ts', 'seq', 'device_ts', 'relative_time_s', 'orig_temp', 'temp_c', 'display_temp', 'error_flag',
        'recv_crc', 'computed_crc', 'crc_ok', 'latency_s', 'masked', 'raw_line'
    ])
# Optional debug log file for firmware debug messages (lines starting with 'DBG:')
DEBUG_LOG_FILE = "serial_debug.log"
debug_log = open(DEBUG_LOG_FILE, 'a', newline='')


def safe_float(s):
    try:
        return float(s)
    except Exception:
        return None


def compute_crc32_hex(s: str) -> str:
    # return 8-char lowercase hex string
    return format(binascii.crc32(s.encode('utf-8')) & 0xFFFFFFFF, '08x')


def update(frame):
    global start_time
    global ema
    # Consume all available queued serial lines and process them
    processed = False
    while True:
        try:
            host_recv, line_bytes = line_queue.get_nowait()
        except queue.Empty:
            break

        processed = True
        host_iso = datetime.datetime.utcfromtimestamp(host_recv).isoformat()

        raw = line_bytes.decode('utf-8', errors='replace').strip()

        # If firmware printed a debug message (e.g., "DBG: raw=..."), handle it
        # separately so it doesn't get interpreted as a temperature sample.
        if 'DBG:' in raw:
            # write debug lines to a separate file and also print to console for visibility
            try:
                debug_log.write(f"{datetime.datetime.utcfromtimestamp(host_recv).isoformat()} {raw}\n")
                debug_log.flush()
            except Exception:
                pass
            # also print compact debug to console
            print(raw)
            # skip processing this line as a temperature sample
            continue
        seq = None
        temp = None
        orig_temp = None
        error_flag = None
        device_ts = None
        recv_crc = None
        computed_crc = None
        crc_ok = None
        latency = None

        fields = [f.strip() for f in raw.split(',') if f.strip() != '']

        # Structured payload: seq,temp,error_flag,device_timestamp,crc
        if len(fields) >= 2:
            # try to interpret structured form; be forgiving about missing fields
            try:
                # If first field looks like an int sequence number
                if len(fields) >= 5:
                    seq = int(fields[0]) if fields[0].isdigit() else None
                    temp = safe_float(fields[1])
                    try:
                        error_flag = bool(int(fields[2]))
                    except Exception:
                        error_flag = False
                    device_ts = safe_float(fields[3])
                    recv_crc = fields[4]
                    payload = ','.join(fields[:4])
                    computed_crc = compute_crc32_hex(payload)
                    crc_ok = (recv_crc.lower() == computed_crc.lower()) if recv_crc else None
                    if device_ts is not None:
                        latency = host_recv - device_ts
                else:
                    # Fallback structured parse (seq,temp) or (temp,error)
                    if len(fields) == 2:
                        # probably legacy temp + extra garbage; treat as temp
                        temp = safe_float(fields[0]) if safe_float(fields[0]) is not None else safe_float(fields[1])
                    elif len(fields) == 3:
                        # maybe seq,temp,error
                        seq = int(fields[0]) if fields[0].isdigit() else None
                        temp = safe_float(fields[1])
                        try:
                            error_flag = bool(int(fields[2]))
                        except Exception:
                            error_flag = False
            except Exception:
                # If structured parse fails, try legacy single-float parse below
                temp = None

        # Legacy single float per line parse
        if temp is None:
            try:
                temp = float(raw)
            except Exception:
                # final fallback: attempt to extract first float token
                for token in raw.replace(';', ',').split(','):
                    try:
                        temp = float(token)
                        break
                    except Exception:
                        continue

        # Keep original parsed value (if any) for logging and unit detection
        if temp is not None:
            orig_temp = temp

        # Heuristic: if value looks like Fahrenheit, convert to Celsius
        temp_c = None
        if orig_temp is not None:
            if AUTO_CONVERT_F_TO_C and (F_TO_C_THRESHOLD is not None) and (orig_temp > F_TO_C_THRESHOLD):
                # plausible Fahrenheit value -> convert
                temp_c = (orig_temp - 32.0) * 5.0 / 9.0
            else:
                temp_c = orig_temp

        # Manage start time and relative time axis
        if start_time is None:
            start_time = host_recv
        rel_time = host_recv - start_time

        # Decide masking and display value BEFORE updating EMA so the EMA is
        # only trained on trusted (unmasked) samples.
        display_temp = None
        masked_flag = 0
        display_for_plot = None
        # allow a quick divisor-based correction for obviously-wrong readings
        temp_proc = temp_c
        auto_scaled = False
        if temp_c is not None and ENABLE_DIVIDE_OVERRIDE and temp_c > DIVIDE_THRESHOLD:
            temp_proc = temp_c / DIVIDE_FACTOR
            auto_scaled = True

        if temp_proc is not None:
            # Determine whether this sample looks out-of-range (use processed value)
            if ENABLE_MASKING and (temp_proc > MAX_DISPLAY_TEMP or temp_proc < MIN_DISPLAY_TEMP):
                masked_flag = 1
                # Prefer the last known good display value
                if last_good_display is not None:
                    display_for_plot = last_good_display
                else:
                    # If EMA exists and is in-range, use it; otherwise clamp
                    if USE_EMA and (ema is not None) and (MIN_DISPLAY_TEMP <= ema <= MAX_DISPLAY_TEMP):
                        display_for_plot = ema
                    else:
                        display_for_plot = max(min(temp_proc, MAX_DISPLAY_TEMP), MIN_DISPLAY_TEMP)
            else:
                # Trusted sample: update EMA (if enabled) and use it for display
                if USE_EMA:
                    if ema is None:
                        ema = temp_proc
                    else:
                        ema = EMA_ALPHA * temp_proc + (1.0 - EMA_ALPHA) * ema
                    display_for_plot = ema
                else:
                    display_for_plot = temp_proc
                # remember as last good value
                last_good_display = display_for_plot

        if display_for_plot is not None:
            # Determine masked display value before appending so the plotted
            # values reflect masking immediately. We still log the original
            # parsed temperature (orig_temp) and the computed temp_c.
            masked = masked_flag
            # display_for_plot already computed above

            x_data.append(rel_time)
            y_data.append(display_for_plot)
            # keep window
            if len(x_data) > MAX_POINTS:
                x_data.pop(0)
                y_data.pop(0)
            line.set_data(x_data, y_data)
            ax.set_xlim(max(0, x_data[0]), x_data[-1] + 1)
            ax.set_ylim(min(y_data) - 1, max(y_data) + 1)

            csv_writer.writerow([
                host_iso,
                seq if seq is not None else '',
                device_ts if device_ts is not None else '',
                f"{rel_time:.6f}",
                f"{orig_temp:.3f}" if orig_temp is not None else '',
                f"{temp_c:.3f}" if temp_c is not None else '',
                f"{display_for_plot:.3f}" if display_for_plot is not None else '',
                int(error_flag) if isinstance(error_flag, bool) else '',
                recv_crc if recv_crc is not None else '',
                computed_crc if computed_crc is not None else '',
                crc_ok if crc_ok is not None else '',
                f"{latency:.6f}" if latency is not None else '',
                int(masked),
                raw
            ])
        # Ensure data flushed to disk reasonably frequently
        log_file.flush()

        # Show warnings for crc mismatch or error flags
        if crc_ok is False:
            print(f"CRC MISMATCH: seq={seq} recv_crc={recv_crc} computed={computed_crc} raw={raw}")
        if error_flag:
            print(f"Device reported error flag in packet seq={seq} raw={raw}")

    # end while consuming queue
    return line,


# Note: blit=True can prevent axis changes (x/y limits) from updating properly.
# Disable blitting so axis autoscaling and x-axis labels update as new data arrives.
ani = FuncAnimation(fig, update, blit=False, interval=100)
plt.show()


def cleanup():
    try:
        if ser and ser.is_open:
            ser.close()
    except Exception:
        pass
    try:
        log_file.close()
    except Exception:
        pass
    try:
        debug_log.close()
    except Exception:
        pass


atexit.register(cleanup)

