"""
SWAN BLE scale reader and live dashboard.

Reads weight directly from the scale's Bluetooth Low Energy notify characteristic,
so the measurement never passes through the vision path. Protocol recovered by
reverse engineering (see the BLE Scale tab in index.html):

    device name : SWAN
    notify UUID : 0000ffb2-0000-1000-8000-00805f9b34fb
    packet      : 8 bytes, weight is little-endian grams
        low  = packet[4]                       # 0..255
        high = packet[5] | (packet[3] & 0xFE)  # 256s (carry may sit in the status byte)
        weight = low + high * 256

A sticky high byte holds the last non-zero carry (the high byte can momentarily
read 0 during a refresh) and is reset only when the load returns to near zero.
A reading is marked STABLE once the last 10 samples vary by no more than 2 g.

Run: python scripts/ble/scale_reader.py   (requires: bleak, matplotlib, numpy)
"""
import asyncio
import threading
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bleak import BleakScanner, BleakClient
import numpy as np

# --- הגדרות ---
NOTIFY_UUID = "0000ffb2-0000-1000-8000-00805f9b34fb"
MAX_HISTORY = 150
STABILITY_WINDOW = 10  # כמה דגימות אחרונות לבדוק ליציבות
STABILITY_THRESHOLD = 2  # סטייה מותרת בגרמים (כדי להיחשב יציב)

# הגדרות סקאלה
MIN_Y_AXIS = 1000
MAX_Y_SCALE_LIMIT = 6000

# צבעים
COLOR_BG = '#121212'
COLOR_GRID = '#333333'
COLOR_LINE = '#00ffcc'
COLOR_FILL = '#00ffcc'
COLOR_STABLE = '#00ffcc'
COLOR_MOVING = '#ff5555'

# --- משתנים גלובליים ---
weight_buffer = deque([0] * MAX_HISTORY, maxlen=MAX_HISTORY)
stability_buffer = deque([0] * STABILITY_WINDOW, maxlen=STABILITY_WINDOW)  # באפר חדש ליציבות

current_weight = 0
current_status = "CONNECTING..."
is_stable_flag = False  # יחושב ע"י התוכנה
sticky_high_byte = 0


# --- לוגיקת בלוטוס ---
def notification_handler(sender, data):
    global current_weight, current_status, is_stable_flag, sticky_high_byte
    hex_data = list(data)
    if len(hex_data) < 8: return

    # פיענוח משקל (V6 Protocol)
    low_byte = hex_data[4]
    current_high_byte = hex_data[5] | (hex_data[3] & 0xFE)
    if current_high_byte > 0: sticky_high_byte = current_high_byte
    if low_byte < 5 and current_high_byte == 0: sticky_high_byte = 0
    final_high = max(current_high_byte, sticky_high_byte)
    weight = low_byte + (final_high * 256)

    # --- חישוב יציבות בתוכנה (Software Stability) ---
    stability_buffer.append(weight)

    # בדיקה: האם יש לנו מספיק דגימות, והאם הפער ביניהן קטן?
    if len(stability_buffer) == STABILITY_WINDOW:
        delta = max(stability_buffer) - min(stability_buffer)
        software_stable = delta <= STABILITY_THRESHOLD
    else:
        software_stable = False

    # עדכון משתנים
    current_weight = weight
    is_stable_flag = software_stable
    current_status = "STABLE" if is_stable_flag else "MOVING..."

    weight_buffer.append(weight)


async def ble_task():
    while True:
        try:
            print("Searching for SWAN Scale...")
            device = await BleakScanner.find_device_by_name("SWAN")
            if device:
                print(f"✅ Connected to {device.name}")
                async with BleakClient(device) as client:
                    await client.start_notify(NOTIFY_UUID, notification_handler)
                    while client.is_connected:
                        await asyncio.sleep(1)
            else:
                await asyncio.sleep(3)
        except Exception:
            await asyncio.sleep(1)


def start_ble_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_task())


# --- לוגיקת GUI (ללא שינוי, רק משתמשת בסטטוס החדש) ---
def update_graph(frame):
    ax.clear()
    data_list = list(weight_buffer)
    x_axis = np.arange(len(data_list))

    ax.plot(x_axis, data_list, color=COLOR_LINE, linewidth=2.5, alpha=0.8)
    ax.fill_between(x_axis, data_list, 0, color=COLOR_FILL, alpha=0.15)
    ax.axhline(y=current_weight, color=COLOR_LINE, linestyle='--', alpha=0.3, linewidth=1)

    current_max = max(data_list) if data_list else 0
    target_ylim = min(max(MIN_Y_AXIS, current_max * 1.2), MAX_Y_SCALE_LIMIT)
    ax.set_ylim(0, target_ylim)
    ax.set_xlim(0, MAX_HISTORY)

    ax.grid(True, color=COLOR_GRID, linestyle=':', linewidth=0.8)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.tick_params(axis='y', colors='white')

    ax.text(0.02, 0.95, 'SWAN IoT Scale // Live Feed', transform=ax.transAxes,
            color='white', fontsize=12, alpha=0.7, weight='bold')

    status_color = COLOR_STABLE if is_stable_flag else COLOR_MOVING

    ax.text(0.5, 0.85, f"{current_weight} g", transform=ax.transAxes,
            ha='center', va='center', fontsize=50, color='white', weight='bold',
            fontname='Arial Rounded MT Bold')

    ax.text(0.5, 0.75, f"STATUS: {current_status}", transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color=status_color, weight='bold',
            bbox=dict(facecolor=COLOR_BG, edgecolor=status_color, boxstyle='round,pad=0.3', alpha=0.8))


if __name__ == "__main__":
    t = threading.Thread(target=start_ble_thread, daemon=True)
    t.start()

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    fig.canvas.manager.set_window_title('SWAN Smart Scale Dashboard V3')

    ani = animation.FuncAnimation(fig, update_graph, interval=30, cache_frame_data=False)
    plt.show()