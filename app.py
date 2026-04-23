import os
import cv2
import numpy as np
import psycopg2
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import threading
from tracker import BYTETracker
from datetime import datetime

app = Flask(__name__)
if os.getenv('DOCKER'):
    app.config['UPLOAD_FOLDER'] = '/app/static/uploads'
    app.config['RESULT_FOLDER'] = '/app/static/results'
else:
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME', 'maritime_tracking')
db_user = os.getenv('DB_USER', 'postgres')
db_password = os.getenv('DB_PASSWORD', 'postgres')

DB_CONFIG = {
    'host': db_host,
    'port': db_port,
    'database': db_name,
    'user': db_user,
    'password': db_password,
    'options': '-c client_encoding=UTF8'
}

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

processing_progress = {}
processing_lock = threading.Lock()
task_results = {}


def init_db_with_retry(max_retries=10, delay=3):
    import time
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            conn.set_client_encoding('UTF8')
            cur = conn.cursor()

            cur.execute('''
                CREATE TABLE IF NOT EXISTS tracking_sessions (
                    id SERIAL PRIMARY KEY,
                    video_name VARCHAR(255),
                    model_type VARCHAR(50),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP DEFAULT NOW(),
                    total_frames INTEGER,
                    total_objects INTEGER,
                    total_detections INTEGER,
                    processing_fps REAL,
                    video_fps REAL,
                    avg_lifetime REAL,
                    avg_confidence REAL,
                    bbox_stability REAL,
                    track_thresh REAL,
                    match_thresh REAL,
                    track_buffer INTEGER,
                    kf_q REAL,
                    kf_r REAL,
                    kf_p REAL,
                    output_filename VARCHAR(255)
                )
            ''')

            # Commit table creation
            conn.commit()

            cur.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'tracking_sessions' AND column_name = 'avg_confidence'
                    ) THEN
                        ALTER TABLE tracking_sessions ADD COLUMN avg_confidence REAL;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'tracking_sessions' AND column_name = 'bbox_stability'
                    ) THEN
                        ALTER TABLE tracking_sessions ADD COLUMN bbox_stability REAL;
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'tracking_sessions' AND column_name = 'track_thresh'
                    ) THEN
                        ALTER TABLE tracking_sessions ADD COLUMN track_thresh REAL;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'tracking_sessions' AND column_name = 'match_thresh'
                    ) THEN
                        ALTER TABLE tracking_sessions ADD COLUMN match_thresh REAL;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'tracking_sessions' AND column_name = 'track_buffer'
                    ) THEN
                        ALTER TABLE tracking_sessions ADD COLUMN track_buffer INTEGER;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'tracking_sessions' AND column_name = 'kf_q'
                    ) THEN
                        ALTER TABLE tracking_sessions ADD COLUMN kf_q REAL;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'tracking_sessions' AND column_name = 'kf_r'
                    ) THEN
                        ALTER TABLE tracking_sessions ADD COLUMN kf_r REAL;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'tracking_sessions' AND column_name = 'kf_p'
                    ) THEN
                        ALTER TABLE tracking_sessions ADD COLUMN kf_p REAL;
                    END IF;
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'tracking_sessions' AND column_name = 'output_filename'
                    ) THEN
                        ALTER TABLE tracking_sessions ADD COLUMN output_filename VARCHAR(255);
                    END IF;
                END $$
            ''')

            # Commit ALTER TABLE changes
            conn.commit()

            cur.execute('''
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'tracking_sessions' AND column_name = 'lost_count'
            ''')
            if cur.fetchone():
                cur.execute('ALTER TABLE tracking_sessions DROP COLUMN lost_count')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS tracks (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES tracking_sessions(id),
                    track_id INTEGER NOT NULL,
                    start_frame INTEGER NOT NULL,
                    end_frame INTEGER NOT NULL,
                    lifetime INTEGER NOT NULL,
                    lifetime_seconds REAL,
                    avg_confidence REAL,
                    min_x1 REAL,
                    min_y1 REAL,
                    max_x2 REAL,
                    max_y2 REAL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            cur.execute('''
                ALTER TABLE tracks 
                ADD COLUMN IF NOT EXISTS min_x1 REAL,
                ADD COLUMN IF NOT EXISTS min_y1 REAL,
                ADD COLUMN IF NOT EXISTS max_x2 REAL,
                ADD COLUMN IF NOT EXISTS max_y2 REAL
            ''')
            
            cur.execute('''
                CREATE TABLE IF NOT EXISTS tracks (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES tracking_sessions(id),
                    track_id INTEGER NOT NULL,
                    frame_number INTEGER NOT NULL,
                    frame_time TIMESTAMP,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    confidence REAL NOT NULL,
                    class_id INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')

            cur.execute('''
                ALTER TABLE tracks
                ADD COLUMN IF NOT EXISTS min_x1 REAL,
                ADD COLUMN IF NOT EXISTS min_y1 REAL,
                ADD COLUMN IF NOT EXISTS max_x2 REAL,
                ADD COLUMN IF NOT EXISTS max_y2 REAL
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES tracking_sessions(id),
                    track_id INTEGER NOT NULL,
                    frame_number INTEGER NOT NULL,
                    frame_time TIMESTAMP,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    confidence REAL NOT NULL,
                    class_id INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')

            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(delay)
    return False


def save_session(video_name, model_type, metrics, task_id, started_at, track_thresh=None, match_thresh=None, track_buffer=None, kf_q=None, kf_r=None, kf_p=None, output_filename=None):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_client_encoding('UTF8')
        cur = conn.cursor()
        
        completed_at = datetime.now()
        
        cur.execute('''
            INSERT INTO tracking_sessions
            (video_name, model_type, total_frames, total_objects, total_detections,
             processing_fps, video_fps, avg_lifetime, avg_confidence, bbox_stability,
             track_thresh, match_thresh, track_buffer, kf_q, kf_r, kf_p, output_filename, started_at, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (
            video_name, model_type,
            int(metrics.get('frames', 0)),
            int(metrics.get('objects', 0)),
            int(metrics.get('detections', 0)),
            float(metrics.get('processing_fps', 0)),
            float(metrics.get('video_fps', 0)),
            float(metrics.get('avg_lifetime', 0)),
            float(metrics.get('avg_confidence', 0)),
            float(metrics.get('bbox_stability', 0)),
            float(track_thresh) if track_thresh is not None else None,
            float(match_thresh) if match_thresh is not None else None,
            int(track_buffer) if track_buffer is not None else None,
            float(kf_q) if kf_q is not None else None,
            float(kf_r) if kf_r is not None else None,
            float(kf_p) if kf_p is not None else None,
            output_filename,
            started_at,
            completed_at
        ))

        session_id = cur.fetchone()[0]

        if metrics.get('track_details'):
            for track in metrics['track_details']:
                cur.execute('''
                    INSERT INTO tracks (session_id, track_id, start_frame, end_frame, lifetime, lifetime_seconds, avg_confidence, min_x1, min_y1, max_x2, max_y2)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    session_id, 
                    int(track['id']), 
                    int(track.get('start_frame', 0)), 
                    int(track.get('end_frame', 0)), 
                    int(track['lifetime']),
                    float(track.get('lifetime_seconds', 0)),
                    float(track.get('avg_confidence', 0)),
                    float(track.get('min_x1', 0)),
                    float(track.get('min_y1', 0)),
                    float(track.get('max_x2', 0)),
                    float(track.get('max_y2', 0))
                ))
        
        if metrics.get('detection_details'):
            for det in metrics['detection_details']:
                cur.execute('''
                    INSERT INTO detections (session_id, track_id, frame_number, frame_time, x1, y1, x2, y2, confidence, class_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    session_id,
                    int(det.get('track_id', 0)),
                    int(det['frame']),
                    det.get('frame_time'),
                    float(det['x1']),
                    float(det['y1']),
                    float(det['x2']),
                    float(det['y2']),
                    float(det['confidence']),
                    int(det.get('class_id', 0))
                ))
        
        conn.commit()
        cur.close()
        conn.close()
        return session_id
    except Exception as e:
        print(f"DB save error: {e}")
        return None


def get_sessions(limit=10):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_client_encoding('UTF8')
        cur = conn.cursor()
        cur.execute('''
            SELECT id, video_name, model_type, total_frames, total_objects, processing_fps,
                   video_fps, avg_lifetime, avg_confidence, bbox_stability,
                   track_thresh, match_thresh, track_buffer, kf_q, kf_r, kf_p, output_filename, completed_at
            FROM tracking_sessions
            ORDER BY completed_at DESC
            LIMIT %s
        ''', (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        return []


def get_session_tracks(session_id):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_client_encoding('UTF8')
        cur = conn.cursor()
        cur.execute('''
            SELECT track_id, start_frame, end_frame, lifetime, avg_confidence 
            FROM tracks 
            WHERE session_id = %s 
            ORDER BY track_id
        ''', (session_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"DB error: {e}")
        return []


def get_session_detections(session_id, limit=500):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_client_encoding('UTF8')
        cur = conn.cursor()
        cur.execute('''
            SELECT track_id, frame_number, x1, y1, x2, y2, confidence, class_id
            FROM detections 
            WHERE session_id = %s 
            ORDER BY frame_number, track_id
            LIMIT %s
        ''', (session_id, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"DB error: {e}")
        return []


def box_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def process_video(input_path, output_path, model_type, task_id, device=None, track_thresh=0.4, match_thresh=0.8, track_buffer=190, kf_q=0.025, kf_r=0.1, kf_p=10.0, output_filename=None):
    global processing_progress

    if device is None or device not in ('cpu', 'cuda'):
        device = get_device()

    try:

        model_map = {
            'yolo8x': 'weights/best_yolo8x.pt',
            'yolo26x': 'weights/best_yolo26x.pt',
            'rtdetr-x': 'weights/best_rtdetr-x.pt',
            'rtdetr-l': 'weights/best_rtdetr-l.pt'
        }
        model_file = model_map.get(model_type, 'weights/best_yolo8x.pt')
        model = YOLO(model_file)
        model.to(device)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Could not open input video: {input_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use default MP4 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        if not out.isOpened():
            print(f"Failed to open video writer for {output_path}")
            raise Exception(f"Could not open video writer for {output_path}")

        tracker = BYTETracker(track_thresh=float(track_thresh), track_buffer=int(track_buffer), match_thresh=float(match_thresh), kf_q=float(kf_q), kf_r=float(kf_r), kf_p=float(kf_p))
        
        frame_num = 0
        total_detections = 0
        unique_track_ids = set()
        track_lifetimes = {}
        track_start_frames = {}
        track_end_frames = {}
        track_confidences = {}
        track_min_coords = {}
        track_max_coords = {}
        all_confidences = []
        prev_track_bboxes = {}
        bbox_iou_sum = 0
        bbox_iou_count = 0
        prev_active_ids = set()
        lost_track_ids = set()
        detection_details = []
        start_time = cv2.getTickCount()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = datetime.now()
            results = model(frame, verbose=False, device=device)[0]
            boxes = results.boxes

            CLASS_NAMES = {0: 'powerboat', 1: 'sailboat', 2: 'ship', 3: 'stationary'}

            detections = []
            if len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                    detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls])
                    all_confidences.append(conf)
                    total_detections += 1

            tracks = tracker.update(detections)

            for track in tracks:
                unique_track_ids.add(track['track_id'])
                track_id = track['track_id']
                
                if track_id not in track_start_frames:
                    track_start_frames[track_id] = track['start_frame']
                track_end_frames[track_id] = track['frame_id']
                track_lifetimes[track_id] = track['frame_id'] - track['start_frame'] + 1
                
                if track_id not in track_confidences:
                    track_confidences[track_id] = []
                track_confidences[track_id].append(track['score'])
                
                tlbr = track['tlbr']
                if track_id not in track_min_coords:
                    track_min_coords[track_id] = [tlbr[0], tlbr[1]]
                    track_max_coords[track_id] = [tlbr[2], tlbr[3]]
                else:
                    track_min_coords[track_id][0] = min(track_min_coords[track_id][0], tlbr[0])
                    track_min_coords[track_id][1] = min(track_min_coords[track_id][1], tlbr[1])
                    track_max_coords[track_id][0] = max(track_max_coords[track_id][0], tlbr[2])
                    track_max_coords[track_id][1] = max(track_max_coords[track_id][1], tlbr[3])
                
                detection_details.append({
                    'track_id': track_id,
                    'frame': frame_num,
                    'frame_time': frame_time,
                    'x1': float(track['tlbr'][0]),
                    'y1': float(track['tlbr'][1]),
                    'x2': float(track['tlbr'][2]),
                    'y2': float(track['tlbr'][3]),
                    'confidence': float(track['score']),
                    'class_id': int(track.get('class', 0))
                })
                tlbr = track['tlbr']
                track_id = track['track_id']
                track_cls = track.get('class', 0)
                if hasattr(track_cls, 'item'):
                    track_cls = int(track_cls.item())
                elif isinstance(track_cls, np.ndarray):
                    track_cls = int(track_cls.flat[0]) if track_cls.size > 0 else 0
                else:
                    track_cls = int(track_cls) if track_cls else 0
                class_name = CLASS_NAMES.get(track_cls, f'class_{track_cls}')
                
                color = ((track_id * 37) % 255, (track_id * 97) % 255, (track_id * 71) % 255)
                cv2.rectangle(frame, (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3])), color, 2)
                
                avg_conf = sum(track['scores']) / len(track['scores']) if track['scores'] else track['score']
                label = f'ID:{track_id} cls:{track_cls} {avg_conf:.2f}'
                
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                label_y = int(tlbr[1]) + text_h + 8
                
                cv2.rectangle(frame, (int(tlbr[0]), label_y - text_h - 2), (int(tlbr[0]) + text_w + 4, label_y + 2), color, -1)
                cv2.putText(frame, label, (int(tlbr[0]) + 2, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                history = track.get('history', [])
                if len(history) > 1:
                    for i in range(1, len(history)):
                        pt1 = (int((history[i-1][0] + history[i-1][2]) / 2), 
                               int((history[i-1][1] + history[i-1][3]) / 2))
                        pt2 = (int((history[i][0] + history[i][2]) / 2), 
                               int((history[i][1] + history[i][3]) / 2))
                        cv2.line(frame, pt1, pt2, color, 2)
                
                if track_id in prev_track_bboxes:
                    prev_box = prev_track_bboxes[track_id]
                    curr_box = track['tlbr']
                    iou = box_iou(prev_box, curr_box)
                    bbox_iou_sum += iou
                    bbox_iou_count += 1
                prev_track_bboxes[track_id] = track['tlbr']
            
            current_active_ids = set(t['track_id'] for t in tracks)
            lost_ids = prev_active_ids - current_active_ids
            lost_track_ids = lost_track_ids | lost_ids
            prev_active_ids = current_active_ids

            out.write(frame)
            frame_num += 1



            progress = int((frame_num / total_frames) * 100)
            with processing_lock:
                processing_progress[task_id] = progress

        cap.release()
        out.release()

        mot_file = export_motchallenge_file(detection_details, output_filename, app.config['RESULT_FOLDER'])
        task_results[task_id]['mot_file'] = f'/static/results/{mot_file}'  # Сохраняем ссылку для API

        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        processing_fps = frame_num / elapsed if elapsed > 0 else 0
        
        avg_lifetime = sum(track_lifetimes.values()) / len(track_lifetimes) if track_lifetimes else 0
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        bbox_stability = float(bbox_iou_sum / bbox_iou_count * 100) if bbox_iou_count > 0 else 0
        
        track_details = []
        for track_id, lifetime in track_lifetimes.items():
            start_frame = track_start_frames.get(track_id, 0)
            end_frame = track_end_frames.get(track_id, 0)
            confidences = track_confidences.get(track_id, [])
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            min_coords = track_min_coords.get(track_id, [0, 0])
            max_coords = track_max_coords.get(track_id, [0, 0])
            lifetime_seconds = round(float(lifetime) / video_fps, 2) if video_fps > 0 else 0
            track_details.append({
                'id': track_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'lifetime': lifetime,
                'lifetime_seconds': lifetime_seconds,
                'avg_confidence': round(float(avg_conf), 3),
                'min_x1': round(float(min_coords[0]), 1),
                'min_y1': round(float(min_coords[1]), 1),
                'max_x2': round(float(max_coords[0]), 1),
                'max_y2': round(float(max_coords[1]), 1)
            })
        
        metrics = {
            'frames': frame_num,
            'objects': len(unique_track_ids),
            'detections': total_detections,
            'processing_fps': round(float(processing_fps), 1),
            'video_fps': int(video_fps),
            'avg_lifetime': round(float(avg_lifetime), 1),
            'avg_confidence': round(float(avg_confidence), 3),
            'bbox_stability': round(float(bbox_stability), 1),
            'track_details': track_details,
            'detection_details': detection_details
        }
        
        task_results[task_id]['metrics'] = metrics
        task_results[task_id]['device'] = device

        try:
            started_at = task_results[task_id].get('started_at', datetime.now())
            session_id = save_session(os.path.basename(input_path), model_type, metrics, task_id, started_at, track_thresh, match_thresh, track_buffer, kf_q, kf_r, kf_p, output_filename)
            task_results[task_id]['db_session_id'] = session_id
        except Exception as e:
            pass

        with processing_lock:
            processing_progress[task_id] = 100

    except Exception as e:
        with processing_lock:
            processing_progress[task_id] = -1

def export_motchallenge_file(detections, output_video_filename, result_folder):
    """Сохраняет детекции в формате MOTChallenge (.txt) рядом с обработанным видео."""
    base_name = os.path.splitext(output_video_filename)[0]
    mot_filename = f"{base_name}.txt"
    mot_path = os.path.join(result_folder, mot_filename)
    
    sorted_dets = sorted(detections, key=lambda x: x['frame'])
    
    with open(mot_path, 'w', encoding='utf-8') as f:
        for det in sorted_dets:
            frame = int(det['frame']) + 1 
            track_id = int(det['track_id'])
            x1, y1 = float(det['x1']), float(det['y1'])
            x2, y2 = float(det['x2']), float(det['y2'])
            w, h = x2 - x1, y2 - y1
            conf = float(det['confidence'])
            cls = int(det.get('class_id', 0))
            visibility = 1.0
            ignore = 0
            
            # Формат: <frame>,<id>,<x1>,<y1>,<w>,<h>,<conf>,<class>,<visibility>,<ignore>
            f.write(f"{frame},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},{cls},{visibility}\n")
            
    return mot_filename


init_db_with_retry()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sessions')
def api_sessions():
    try:
        sessions = get_sessions(limit=20)
        return jsonify({'success': True, 'sessions': [
            {'id': s[0], 'video_name': str(s[1]), 'model_type': str(s[2]), 'frames': s[3],
             'objects': s[4], 'fps': s[5], 'video_fps': s[6], 'avg_lifetime': s[7],
             'avg_confidence': s[8], 'bbox_stability': s[9],
             'track_thresh': s[10], 'match_thresh': s[11], 'track_buffer': s[12],
             'kf_q': s[13], 'kf_r': s[14], 'kf_p': s[15], 'output_filename': str(s[16]), 'completed_at': str(s[17])}
            for s in sessions
        ]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sessions/<int:session_id>')
def api_session_detail(session_id):
    try:
        tracks = get_session_tracks(session_id)
        detections = get_session_detections(session_id, limit=200)
        return jsonify({'success': True, 'tracks': [
            {'track_id': t[0], 'start_frame': t[1], 'end_frame': t[2], 'lifetime': t[3], 'avg_confidence': t[4]} 
            for t in tracks
        ], 'detections': [
            {'track_id': d[0], 'frame': d[1], 'x1': d[2], 'y1': d[3], 'x2': d[4], 'y2': d[5], 'conf': d[6], 'class_id': d[7]}
            for d in detections
        ]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get-result/<task_id>')
def get_result(task_id):
    result = task_results.get(task_id, {})
    result_video = result.get('result_video')
    metrics = result.get('metrics', {})
    device = result.get('device', 'cpu')
    if result_video:
        return jsonify({'success': True, 'result_video': result_video, 'metrics': metrics, 'device': device})
    return jsonify({'error': 'Task not found'}), 404


@app.route('/progress/<task_id>')
def progress(task_id):
    def generate():
        while True:
            with processing_lock:
                prog = processing_progress.get(task_id, 0)
            
            if prog == -1:
                yield 'data: error\n\n'
                break
            elif prog == 100:
                yield 'data: complete\n\n'
                break
            else:
                yield f'data: {prog}\n\n'
            
            import time
            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    model_type = request.form.get('model', 'yolo')
    device = request.form.get('device', None)
    track_thresh = request.form.get('track_thresh', '0.4')
    match_thresh = request.form.get('match_thresh', '0.8')
    track_buffer = request.form.get('track_buffer', '190')
    kf_q = request.form.get('kf_q', '0.025')
    kf_r = request.form.get('kf_r', '0.1')
    kf_p = request.form.get('kf_p', '10.0')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        import uuid
        task_id = str(uuid.uuid4())
        
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        base, _ = os.path.splitext(filename)
        output_filename = f'output_{task_id}_{base}.mp4'
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

        started_at = datetime.now()
        
        try:
            thread = threading.Thread(target=process_video, args=(input_path, output_path, model_type, task_id, device, track_thresh, match_thresh, track_buffer, kf_q, kf_r, kf_p, output_filename))
            thread.start()
            
            timestamp = int(datetime.now().timestamp())
            timestamp = int(datetime.now().timestamp())
            result_video_url = f'/static/results/{output_filename}?t={timestamp}'
            task_results[task_id] = {'result_video': result_video_url, 'started_at': started_at}

            return jsonify({
                'success': True,
                'task_id': task_id,
                'result_video': result_video_url
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
