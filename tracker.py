import numpy as np
from scipy.optimize import linear_sum_assignment
from enum import Enum

def iou_distance(atlbrs, btlbrs):
    """Вычисляет матрицу расстояний 1-IoU для двух наборов bbox [x1,y1,x2,y2]."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)))
    
    atlbrs = np.asarray(atlbrs)
    btlbrs = np.asarray(btlbrs)
    
    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
    
    lt = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
    rb = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b - inter
    iou = inter / (union + 1e-6)
    return 1.0 - iou

def linear_assignment(cost_matrix, thresh):
    """Венгерский алгоритм с пороговым отсечением."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])
    
    matches_a, matches_b = linear_sum_assignment(cost_matrix)
    matches = np.column_stack((matches_a, matches_b))
    
    # Фильтрация по порогу расстояния
    valid_mask = np.array([cost_matrix[i, j] <= thresh for i, j in matches])
    matches = matches[valid_mask]
    
    unmatched_a = np.setdiff1d(np.arange(cost_matrix.shape[0]), matches[:, 0])
    unmatched_b = np.setdiff1d(np.arange(cost_matrix.shape[1]), matches[:, 1])
    return matches, unmatched_a, unmatched_b

class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class KalmanFilter:
    """8-состояний фильтр Калмана: [cx, cy, aspect_ratio, height, vx, vy, va, vh]"""
    def __init__(self, q=0.025, r=0.1, p=10.0):
        print(f"Initializing KalmanFilter with Q={q}, R={r}, P={p}")
        self._dim_x = 8
        self._dim_z = 4
        # Матрица перехода (постоянная скорость)
        self._F = np.eye(8)
        self._F[0, 4] = self._F[1, 5] = self._F[2, 6] = self._F[3, 7] = 1.0
        # Матрица измерений
        self._H = np.eye(4, 8)
        # Ковариации
        self._Q = np.eye(8) * q  # Процессный шум
        self._R = np.eye(4) * r  # Шум измерений
        self._P = np.eye(8) * p  # Начальная неопределенность
        print(f"KalmanFilter initialized: Q shape={self._Q.shape}, R shape={self._R.shape}, P shape={self._P.shape}")

    def init_state(self, tlbr):
        """Инициализация состояния из bbox [x1,y1,x2,y2]."""
        x1, y1, x2, y2 = tlbr
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        a = w / h if h > 0 else 1.0
        mean = np.array([cx, cy, a, h, 0, 0, 0, 0]).reshape(-1, 1)
        return mean, self._P.copy()

    def predict(self, mean, cov):
        """Предсказание следующего состояния."""
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + self._Q
        return mean, cov

    def update(self, mean, cov, z):
        """Коррекция состояния по новому измерению."""
        y = z - self._H @ mean
        S = self._H @ cov @ self._H.T + self._R
        K = cov @ self._H.T @ np.linalg.inv(S)
        mean = mean + K @ y
        I = np.eye(self._dim_x)
        cov = (I - K @ self._H) @ cov
        return mean, cov

    def convert_to_z(self, tlbr):
        """Преобразование bbox в вектор измерений [cx, cy, a, h]."""
        x1, y1, x2, y2 = tlbr
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        a = w / h if h > 0 else 1.0
        return np.array([cx, cy, a, h]).reshape(-1, 1)

    def convert_to_tlbr(self, mean):
        """Преобразование состояния обратно в bbox [x1,y1,x2,y2]."""
        cx, cy, a, h = mean[:4].flatten()
        w = a * h
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
        return [float(x1), float(y1), float(x2), float(y2)]

class STrack:
    """Класс трека с состоянием фильтра Калмана."""
    def __init__(self, tlbr, score, class_id, frame_id, kf):
        self._tlbr = np.asarray(tlbr, dtype=float)
        self.score = score
        self.class_id = int(class_id)
        self.state = TrackState.New
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.track_id = 0
        self.history = [tlbr]
        self.scores = [score]
        self.mean, self.cov = kf.init_state(tlbr)

    @property
    def tlbr(self):
        """Возвращает предсказанный bbox из фильтра Калмана."""
        return self._tlbr.tolist()

    def predict(self, kf):
        """Предсказание позиции на следующий кадр."""
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self._tlbr = np.asarray(kf.convert_to_tlbr(self.mean), dtype=float)

    def update(self, kf, new_tlbr, new_score):
        """Обновление трека новым детектом."""
        self.mean, self.cov = kf.update(self.mean, self.cov, kf.convert_to_z(new_tlbr))
        self._tlbr = np.asarray(new_tlbr, dtype=float)
        self.score = new_score
        self.state = TrackState.Tracked
        self.frame_id = kf.frame_id if hasattr(kf, 'frame_id') else self.frame_id
        self.history.append(self.tlbr)
        self.scores.append(self.score)
        if len(self.history) > 30:
            self.history = self.history[-30:]
            self.scores = self.scores[-30:]

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def re_activate(self, new_tlbr, new_score, frame_id, kf):
        """Повторная активация потерянного трека."""
        self.update(kf, new_tlbr, new_score)
        self.frame_id = frame_id
        self.state = TrackState.Tracked

class BYTETracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30,
                 kf_q=0.025, kf_r=0.1, kf_p=10.0):
        self.kf = KalmanFilter(q=kf_q, r=kf_r, p=kf_p)
        self.tracked_stracks = []
        self.lost_stracks = []
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.track_id_count = 0

    def next_id(self):
        self.track_id_count += 1
        return self.track_id_count

    def update(self, detections):
        """
        detections: список [x1, y1, x2, y2, score, class_id]
        returns: список словарей с треками (совместимо с app.py)
        """
        self.frame_id += 1
        self.kf.frame_id = self.frame_id
        
        # 1. Разделение детекций на high/low confidence
        dets = np.array(detections) if len(detections) > 0 else np.empty((0, 6))
        dets_high = dets[dets[:, 4] >= self.track_thresh]
        dets_low = dets[(dets[:, 4] < self.track_thresh) & (dets[:, 4] >= 0.1)]
        
        # 2. Предсказание всех активных треков (Kalman)
        for t in self.tracked_stracks:
            t.predict(self.kf)
        for t in self.lost_stracks:
            t.predict(self.kf)
        
        activated, lost = [], []
        
        # 3. Первый раунд: Tracked vs High-Confidence detections
        if len(dets_high) > 0 and len(self.tracked_stracks) > 0:
            dists = iou_distance([t.tlbr for t in self.tracked_stracks], dets_high[:, :4])
            matches, u_tracked, u_dets_high = linear_assignment(dists, self.match_thresh)
            
            for it, idet in matches:
                self.tracked_stracks[it].update(self.kf, dets_high[idet, :4].tolist(), dets_high[idet, 4])
                activated.append(self.tracked_stracks[it])
            for i in u_tracked:
                self.tracked_stracks[i].mark_lost()
                lost.append(self.tracked_stracks[i])
        else:
            for t in self.tracked_stracks:
                t.mark_lost()
                lost.append(t)
            u_dets_high = np.arange(len(dets_high))
            
        # 4. Второй раунд: Lost tracks vs Low-Confidence detections
        if len(dets_low) > 0 and len(self.lost_stracks) > 0:
            dists = iou_distance([t.tlbr for t in self.lost_stracks], dets_low[:, :4])
            matches, u_lost, u_dets_low = linear_assignment(dists, self.match_thresh)
            for it, idet in matches:
                self.lost_stracks[it].re_activate(dets_low[idet, :4].tolist(), dets_low[idet, 4], self.frame_id, self.kf)
                activated.append(self.lost_stracks[it])
        else:
            u_dets_low = np.arange(len(dets_low))
            
        # 5. Третий раунд: Оставшиеся High-Conf vs Оставшиеся Lost (восстановление)
        if len(u_dets_high) > 0 and len(self.lost_stracks) > len(activated):
            remaining_lost = [t for t in self.lost_stracks if t.state == TrackState.Lost]
            if len(remaining_lost) > 0:
                dists = iou_distance([t.tlbr for t in remaining_lost], dets_high[u_dets_high, :4])
                matches, _, _ = linear_assignment(dists, self.match_thresh)
                for it, idet in matches:
                    remaining_lost[it].re_activate(
                        dets_high[u_dets_high][idet, :4].tolist(), 
                        dets_high[u_dets_high][idet, 4], 
                        self.frame_id, self.kf
                    )
                    activated.append(remaining_lost[it])
                    
        # 6. Инициализация новых треков из оставшихся high-conf детекций
        matched_ids = {t.track_id for t in activated}
        for i in u_dets_high:
            if dets_high[i, 4] >= self.track_thresh:
                track = STrack(
                    dets_high[i, :4].tolist(), 
                    dets_high[i, 4], 
                    dets_high[i, 5], 
                    self.frame_id, 
                    self.kf
                )
                track.track_id = self.next_id()
                track.state = TrackState.Tracked
                activated.append(track)
                
        # 7. Очистка старых потерянных треков
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.buffer_size:
                t.mark_removed()
                
        # Обновление списков треков
        self.tracked_stracks = [t for t in activated if t.state == TrackState.Tracked]
        self.lost_stracks = [t for t in self.lost_stracks + lost if t.state == TrackState.Lost]
        
        # Формат вывода, совместимый с app.py
        return [
            {
                'track_id': t.track_id,
                'tlbr': t.tlbr,
                'score': t.score,
                'class': t.class_id,
                'state': t.state,
                'frame_id': self.frame_id,
                'start_frame': t.start_frame,
                'history': t.history,
                'scores': t.scores
            } for t in self.tracked_stracks
        ]