import sys
from threading import Lock
from threading import Thread
import os
import threading
import socket
import struct
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import paramiko
from collections import deque
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import pandas as pd
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch.amp import autocast
from PIL import Image, ImageTk
from scipy.ndimage import uniform_filter1d

# =============================================================================
# Configuration Constants
# =============================================================================
HOST_IP = "192.168.128.1"
DATA_PORT = 61231
SSH_PORT = 22
SSH_USER = "root"
SSH_PASS = "root"
BUFFER_MAX_BLOCKS = 200
ADC_RAW_SIZE = 25000


# Utility function
def current_millis():
    return int(time.time() * 1000)


# =============================================================================
# Module 1: Sensor Acquisition
# =============================================================================
class RedPitayaSensor:
    def __init__(self, log_callback=None):
        self.host = HOST_IP
        self.port = DATA_PORT
        self.server_addr = (self.host, self.port)
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.header_len = None
        self.total_blocks = 0
        self.start_time = 0
        self.running = False
        self.data_buffer = deque(maxlen=BUFFER_MAX_BLOCKS)
        self.log_callback = log_callback
        self.on_receive = None
        self.lock = threading.Lock()
        self.prev_max = None
        self.movement_threshold = 300
        self.prev_max_index = None
        self.max_index_buffer = deque(maxlen=5)

    def set_logger(self, logger):
        self.log_callback = logger

    def set_on_receive(self, callback):
        self.on_receive = callback

    def log(self, msg):
        text = f"[Sensor] {msg}"
        if self.log_callback:
            self.log_callback(text)

    def _ssh_exec(self, cmd):
        self.ssh_client.connect(self.host, SSH_PORT, SSH_USER, SSH_PASS)
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        self.ssh_client.close()
        if err:
            self.log(f"SSH error: {err}")
        return out

    def start_acquisition(self):
        self.log("Starting Direct Memory Access acquisition...")
        self._ssh_exec("cd /usr/RedPitaya/Examples/C && nohup ./dma_with_udp &")
        time.sleep(1)
        self.udp_sock.sendto(b"-i 1", self.server_addr)
        pkt = self.udp_sock.recv((ADC_RAW_SIZE + 6) * 4)
        self.header_len = int(struct.unpack('@f', pkt[:4])[0])
        self.total_blocks = int(struct.unpack('@f', pkt[8:12])[0])
        self.start_time = current_millis()
        self.log(f"Header synced: len={self.header_len}, blocks={self.total_blocks}")

    def stop_acquisition(self):
        self.log("Stopping DMA acquisition...")
        pid = self._ssh_exec("pidof dma_with_udp")
        if pid:
            self._ssh_exec(f"kill {pid}")
        self.running = False

    def read_block(self):
        self.udp_sock.sendto(b"-a 1", self.server_addr)
        pkt = self.udp_sock.recv((ADC_RAW_SIZE + 6) * 4)
        receive_time = current_millis()  # PC timestamp in ms
        data = [v[0] for v in struct.iter_unpack('@h', pkt[self.header_len:])]
        return receive_time, data

    def stream_data(self):
        self.running = True
        self.log("Sensor streaming thread started.")
        while self.running:
            for _ in range(self.total_blocks):
                if not self.running:
                    break
                receive_time, data = self.read_block()
                rec = {'timestamp': receive_time, 'data': data}
                self.data_buffer.append(rec)
                if self.on_receive:
                    self.on_receive(rec)
            time.sleep(0)
        self.log("Sensor streaming thread stopped.")

    def detect_sensor_movement(self, data):
        smoothed = uniform_filter1d(data, size=5)
        current_max_index = int(np.argmax(smoothed))
        self.max_index_buffer.append(current_max_index)

        if len(self.max_index_buffer) < self.max_index_buffer.maxlen:
            return "unknown"

        deltas = np.diff(self.max_index_buffer)
        avg_delta = np.mean(deltas)

        if avg_delta < -self.movement_threshold:
            return "approaching"
        elif avg_delta > self.movement_threshold:
            return "moving_away"
        else:
            return "standing_still"


# =============================================================================
# Module 2: Plotting Controls
# =============================================================================
class PlotManager:
    def __init__(self, figure, axis):
        self.fig = figure
        self.ax = axis
        self.region = None

    def plot_overlay_signals(self, data, realtime=False):
        """
        Overlays multiple signals without vertical offset and applies fading for older ones.
        Expects 'data' shape = (samples, num_signals)
        """
        self.ax.clear()
        data = np.array(data)

        if self.region:
            xmin, xmax = self.region
            data = data[int(xmin):int(xmax)]

        num_samples, num_signals = data.shape

        alphas = np.linspace(0.5, 1.0, num_signals)  # Older = 0.5, Newer = 1.0

        for i in range(num_signals):
            self.ax.plot(range(num_samples), data[:, i], label=f'Signal {i + 1}', alpha=alphas[i])

        self.ax.set_title("ADC Raw Data (Last 2 Signals - Overlayed)")
        self.ax.set_xlabel("Samples [-]")
        self.ax.set_ylabel("Decimal Values")
        self.ax.grid(True)
        
        self.fig.tight_layout()
        self.fig.canvas.draw()


# =============================================================================
# Module 3: Save Counter Manager
# =============================================================================
class SaveManager:
    def __init__(self, save_folder, logger):
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)
        self.logger = logger
        self.target = 0
        self.saved = 0

    def set_target(self, n):
        self.target = n
        self.saved = 0
        self.logger(f"Save target set to {n}")

    def try_save(self, rec):
        if self.saved < self.target:
            folder = os.path.join(self.save_folder, "raw_signals")
            os.makedirs(folder, exist_ok=True)
            f_name = f"signal_{rec['timestamp']}.csv"
            pd.DataFrame(rec['data'], columns=['raw_adc']).to_csv(os.path.join(folder, f_name), index=False)
            self.saved += 1
            self.logger(f"Saved raw signal {self.saved}/{self.target}")
            return True
        return False


# =============================================================================
# Module 4: Vision & Tracking
# =============================================================================
class SegmentationModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT').eval().to(self.device)

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.coco_colors = np.array([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]
        ], dtype=np.uint8)
        self.output_predictions = None

    def process_frame(self, frame, tracker, alert_callback, frame_center):
        frame_resized = cv2.resize(frame, (640, 480))
        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        with autocast('cuda'):
            input_tensor = self.preprocess(image_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]

        self.output_predictions = output.argmax(0).byte().cpu().numpy()
        person_mask = (self.output_predictions == 15).astype(np.uint8) * 255

        segmented_image_bgr = tracker.track_person(person_mask, frame, alert_callback, frame_center)

        segmented_image = self.coco_colors[self.output_predictions]
        segmented_image_resized = cv2.resize(segmented_image, (frame.shape[1], frame.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
        segmented_image_bgr = cv2.cvtColor(segmented_image_resized, cv2.COLOR_RGB2BGR)

        return segmented_image_bgr


class PersonTracker:
    def __init__(self, frame_width, label_mgr, log_callback=None):
        self.frame_width = frame_width
        self.log_callback = log_callback or (lambda msg: print(msg))
        self.person_in_frame = False
        self.label_mgr = label_mgr
        self.last_center_x = None
        self.lock = Lock()
        self.direction = None
        self.area_buffer = deque(maxlen=5)
        self.motion_state = "unknown"
        self.detection_time = None
        self.motion_history = deque(maxlen=50)

    def track_person(self, person_mask, frame, alert_callback, frame_center):
        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        person_detected = False
        largest_area = 0
        largest_contour = None
        self.direction = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue
            if area > largest_area:
                largest_area = area
                largest_contour = contour

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            person_detected = True
            self.last_center_x = center_x
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.area_buffer.append(largest_area)

        if len(self.area_buffer) < self.area_buffer.maxlen:
            self.motion_state = "unknown"
        else:
            prev_avg = sum(list(self.area_buffer)[:-1]) / (self.area_buffer.maxlen - 1)
            latest = self.area_buffer[-1]

            if latest > prev_avg * 1.02:
                self.motion_state = "approaching"
            elif latest < prev_avg * 0.98:
                self.motion_state = "moving_away"
            else:
                self.motion_state = "standing_still"

            self.motion_state_time = current_millis()
            self.motion_history.append((self.motion_state_time, self.motion_state))

            # alert_callback(f"Person is {self.motion_state}")

        if person_detected:
            self.detection_time = current_millis()

        if person_detected:
            if not self.person_in_frame:
                self.direction = "left" if center_x < frame_center else "right"
                alert_callback(f"Person entering from {self.direction}")

                if self.label_mgr:
                    self.label_mgr.entry(f"{self.direction}_entry", self.detection_time)

                self.person_in_frame = True
        else:
            if self.person_in_frame:
                self.direction = "left_exit" if self.last_center_x < frame_center else "right_exit"
                exit_dir = self.direction.split('_')[0]
                alert_callback(f"Person exiting to {exit_dir}")

                if self.label_mgr:
                    self.label_mgr.exit(self.direction)

                self.person_in_frame = False

        return frame, self.direction, person_detected  # Added direction return


# =============================================================================
# Module 5: Sync & Label Manager
# =============================================================================
class LabelManager:
    def __init__(self, base_folder, sensor, logger):
        self.base_folder = base_folder
        os.makedirs(base_folder, exist_ok=True)
        self.sensor = sensor
        self.logger = logger
        self.left_right_buffer = []
        self.entry_dir = None
        self.target_time = None

    def entry(self, direction, detection_time):
        self.entry_dir = direction.replace('_entry', '')
        self.target_time = detection_time + 1000
        self.left_right_buffer = []

    def packet(self, rec):
        if self.entry_dir and rec['timestamp'] > self.target_time:
            self.left_right_buffer.append(rec)

    def exit(self, direction):
        exit_dir = direction.replace('_exit', '')
        folder = os.path.join(self.base_folder, f"entry_{self.entry_dir}_exit_{exit_dir}")
        os.makedirs(folder, exist_ok=True)
        for packet in self.left_right_buffer:
            f_name = f"{packet['timestamp']}.csv"
            pd.DataFrame(packet['data'], columns=['raw_adc']).to_csv(os.path.join(folder, f_name), index=False)
        self.logger(f"Saved {len(self.left_right_buffer)} packets to {folder}")
        self.left_right_buffer = []
        self.entry_dir = None
        self.target_time = None

    def handle_approach_away(self, rec, tracker):
        # Sensor movement detection
        signal_movement = self.sensor.detect_sensor_movement(rec['data'])
        sensor_time = rec['timestamp']

        # Camera movement detection: find recent motion from tracker buffer
        if tracker and tracker.motion_history:
            recent_motions = [state for t, state in tracker.motion_history if sensor_time - 1300 <= t <= sensor_time]
            cam_movement = recent_motions[-1] if recent_motions else "unknown"
        else:
            cam_movement = "unknown"

        # Only save if both agree and are not unknown
        if signal_movement == cam_movement and signal_movement != "unknown":
            folder = os.path.join(self.base_folder, "movement", signal_movement)
            os.makedirs(folder, exist_ok=True)

            f_name = f"{int(rec['timestamp'])}.csv"
            pd.DataFrame(rec['data'], columns=['raw_adc']).to_csv(os.path.join(folder, f_name), index=False)
            self.logger(f"[Approach/Away] Both detected {signal_movement.upper()}, saved {f_name}")
        else:
            self.logger(f"[Approach/Away] Sensor: {signal_movement}, Camera: {cam_movement}")


# =============================================================================
# Module 6: GUI Application
# =============================================================================
class AppGUI:
    def __init__(self, root):
        self.root = root

        self.canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        self._init_modules()
        self._build_ui()
        self.video_frame = None

    def _init_modules(self):
        # Sensor
        self.sensor = RedPitayaSensor(log_callback=self.log)
        self.sensor.app = self  # give sensor access to the full app (for tracker access)

        self.sensor.set_on_receive(self.on_sensor_receive)

        # Plot
        self.fig = Figure(figsize=(5, 3))
        self.ax = self.fig.add_subplot(111)
        self.plot_mgr = PlotManager(self.fig, self.ax)

        # Save
        self.save_mgr = SaveManager(os.path.abspath('saved_signals'), self.log)

        # Vision
        self.seg_model = SegmentationModel()
        self.tracker = None
        self.person_detected = False

        # Label Manager
        self.label_mgr = LabelManager(
            base_folder=os.path.abspath('labeled_data'),
            sensor=self.sensor,
            logger=self.log
        )

        # State
        self.sensor_thread = None
        self.video_thread = None
        self.running = False
        self.cap = None

    def log(self, message):
        self.log_box.config(state='normal')
        self.log_box.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state='disabled')
        self.status_var.set(message)

    def _build_ui(self):
        # Controls frame
        ctrl = ttk.Frame(self.scrollable_frame, padding=5)
        ctrl.pack(fill='x')
        # Sensor buttons
        ttk.Button(ctrl, text="Start Sensor", command=self.start_sensor).pack(side='left')
        ttk.Button(ctrl, text="Stop Sensor", command=self.stop_sensor).pack(side='left', padx=5)
        # Realtime & region
        self.realtime_var = tk.BooleanVar()
        ttk.Checkbutton(ctrl, text="Realtime", variable=self.realtime_var,
                        command=self.toggle_realtime).pack(side='left', padx=8)

        # Save controls
        ttk.Label(ctrl, text="Save signals:").pack(side='left', padx=(10, 2))
        self.save_entry = ttk.Entry(ctrl, width=5);
        self.save_entry.insert(0, '0');
        self.save_entry.pack(side='left')
        ttk.Button(ctrl, text="Set Save", command=self.set_save).pack(side='left', padx=4)

        # Received & saved labels
        ttk.Label(ctrl, text="Received:").pack(side='left', padx=(10, 2))
        self.received_var = tk.IntVar();
        ttk.Label(ctrl, textvariable=self.received_var).pack(side='left')
        ttk.Label(ctrl, text="Saved:").pack(side='left', padx=(10, 2))
        self.saved_var = tk.IntVar();
        ttk.Label(ctrl, textvariable=self.saved_var).pack(side='left')
        # Camera buttons
        ttk.Button(ctrl, text="Start Cam", command=self.start_camera).pack(side='left', padx=8)
        ttk.Button(ctrl, text="Stop Cam", command=self.stop_camera).pack(side='left')

        # Folder selectors
        ttk.Label(ctrl, text="Raw Save Folder:").pack(side='left', padx=(10, 2))
        self.raw_folder_entry = ttk.Entry(ctrl, width=20);
        self.raw_folder_entry.insert(0, os.path.abspath('saved_signals'));
        self.raw_folder_entry.pack(side='left')
        ttk.Button(ctrl, text="Browse", command=self.browse_raw).pack(side='left', padx=4)
        ttk.Label(ctrl, text="Label Save Folder:").pack(side='left', padx=(10, 2))
        self.label_folder_entry = ttk.Entry(ctrl, width=20);
        self.label_folder_entry.insert(0, os.path.abspath('labeled_data'));
        self.label_folder_entry.pack(side='left')
        ttk.Button(ctrl, text="Browse", command=self.browse_label).pack(side='left', padx=4)

        ttk.Label(ctrl, text="Status:").pack(side='left', padx=(10, 2))
        self.status_var = tk.StringVar(master=self.scrollable_frame, value='Idle');
        ttk.Label(ctrl, textvariable=self.status_var).pack(side='left')

        # ===================== Combined Frame ============================
        video_plot_frame = ttk.Frame(self.scrollable_frame)
        video_plot_frame.pack(fill='both', expand=True)

        # Plot (left)
        plot_frame = ttk.Frame(video_plot_frame)
        plot_frame.pack(side='left', padx=5, pady=5)

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.plot_canvas.draw()
        plot_widget = self.plot_canvas.get_tk_widget()
        plot_widget.config(width=640, height=480)  # match camera size
        plot_widget.pack()

        # Camera (right)
        cam_frame = ttk.Frame(video_plot_frame)
        cam_frame.pack(side='left', padx=5, pady=5)

        self.video_canvas = tk.Canvas(cam_frame, width=640, height=480, bg='black')
        self.video_canvas.pack()

        # Switch button under both
        self.mode_var = tk.StringVar(value="orthogonal")
        self.switch_button = ttk.Button(self.scrollable_frame, text="Current mode: Left/Right",
                                        command=self.toggle_mode)
        self.switch_button.pack(padx=10, pady=10)

        # Log console
        self.log_box = scrolledtext.ScrolledText(self.scrollable_frame, height=8, state='disabled')
        self.log_box.pack(fill='both', expand=True, padx=5, pady=5)

    # ================= Sensor Handlers ==================
    def on_sensor_receive(self, rec):
        # Update count
        self.received_var.set(self.received_var.get() + 1)

        # Try SaveManager save (if active)
        self.save_mgr.try_save(rec)

        mode = self.mode_var.get()

        # Left/Right
        if mode == "orthogonal":
            # Save while person is detected
            if self.person_detected:
                self.label_mgr.packet(rec)

        # APPROACH/MOVE AWARE
        elif mode == "sensor":
            self.label_mgr.handle_approach_away(rec, self.tracker)

    def start_sensor(self):
        try:
            self.sensor.start_acquisition()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start sensor: {e}")
            return

        self.sensor_thread = threading.Thread(target=self.sensor.stream_data, daemon=True)
        self.sensor_thread.start()
        self.status_var.set('Sensor Running')

    def stop_sensor(self):
        self.sensor.stop_acquisition()
        self.status_var.set('Sensor Stopped')

    # =============== Plot Handlers =====================
    def toggle_realtime(self):
        if self.realtime_var.get():
            self.update_plot(realtime=True)
        else:
            pass

    # Plot the signal
    def update_plot(self, realtime=False):
        if not realtime:
            # Plot once
            records = list(self.sensor.data_buffer)[-2:]
            if not records:
                return
            signals = [rec['data'] for rec in records]
            data = np.stack(signals, axis=0).T
            self.plot_mgr.plot_overlay_signals(data, realtime=False)
        else:
            self._update_plot_realtime_loop()

    def _update_plot_realtime_loop(self):
        if not self.realtime_var.get():
            return  # Stop if user unchecks the box

        records = list(self.sensor.data_buffer)[-2:]
        if records:
            signals = [rec['data'] for rec in records]
            data = np.stack(signals, axis=0).T
            self.plot_mgr.plot_overlay_signals(data, realtime=False)

        # Repeat after 200ms
        self.root.after(200, self._update_plot_realtime_loop)

    # ============= Save Handlers =======================
    def set_save(self):
        try:
            n = int(self.save_entry.get())
            self.save_mgr.set_target(n)
            self.status_var.set(f"Save {n} signals")
        except:
            messagebox.showerror("Error", "Invalid number")

    def browse_raw(self):
        path = filedialog.askdirectory()
        if path:
            self.save_mgr.save_folder = path  # âœ… Correct â€” SaveManager handles raw signals
            self.raw_folder_entry.delete(0, tk.END)
            self.raw_folder_entry.insert(0, path)
            self.log(f"Raw save folder set to: {path}")

    def browse_label(self):
        path = filedialog.askdirectory()
        if path:
            self.label_mgr.base_folder = path  # âœ… FIXED â€” LabelManager handles labeled data
            self.label_folder_entry.delete(0, tk.END)
            self.label_folder_entry.insert(0, path)
            self.log(f"Label save folder set to: {path}")

    def toggle_mode(self):
        current_mode = self.mode_var.get()
        if current_mode == "orthogonal":
            self.mode_var.set("sensor")
            self.switch_button.config(text="Current mode: Approach/Away")
            self.log("Switched to Approach/Move Away Mode")
        else:
            self.mode_var.set("orthogonal")
            self.switch_button.config(text="Current mode: Left/Right")
            self.log("Switched to Camera Left/Right Movement")

    # ============= Camera Handlers =====================
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to initialize camera")
            return

        # Get actual frame dimensions
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.tracker = PersonTracker(frame_width, label_mgr=self.label_mgr)

        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        self.status_var.set('Camera Running')

    def stop_camera(self):
        if self.cap:
            self.cap.release()
        self.status_var.set('Camera Stopped')
        self.video_canvas.delete("all")

    def video_loop(self):
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame through segmentation and tracking
            segmented_image_bgr = self.seg_model.process_frame(
                frame,
                self.tracker,
                alert_callback=self.log,
                frame_center=frame.shape[1] // 2
            )

            person_mask = (self.seg_model.output_predictions == 15).astype(np.uint8) * 255

            # Track person and get direction
            tracked_frame, direction, self.person_detected = self.tracker.track_person(
                person_mask=person_mask,  # Use the mask from segmentation
                frame=segmented_image_bgr,
                alert_callback=self.log,
                frame_center=frame.shape[1] // 2
            )

            # Convert and display frame
            img = cv2.cvtColor(tracked_frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            self.display_frame(img)

            time.sleep(0.03)

    def display_frame(self, img):
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)

        def update_gui():
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor='nw', image=imgtk)
            self.video_canvas.image = imgtk  # keep reference to avoid garbage collection

        self.root.after(0, update_gui)


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1200x1200')
    app = AppGUI(root)
    root.mainloop()
