"""
Redis camera sync service for area_monitoring (polling + ROI update).

Extends BaseRedisSync to add ROI parsing and on_roi_update callback.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from apps.common.redis_sync_base import BaseRedisSync, BaseCameraConfig

logger = logging.getLogger(__name__)


def circle_to_polygon(center_x: float, center_y: float, radius: float, 
                      num_points: int = 32, config_width: int = 1920, 
                      config_height: int = 1080) -> List[List[int]]:
    """Convert circle to polygon (approximation).
    
    Args:
        center_x, center_y: Normalized coordinates (0-1) or pixel coordinates
        radius: Normalized radius (0-1) or pixel radius
        num_points: Number of points to approximate circle (default 32)
        config_width, config_height: For normalization if needed
    
    Returns:
        List of polygon points in pixel coordinates
    """
    # Check if normalized (0-1) or pixel
    is_normalized = center_x <= 1.0 and center_y <= 1.0 and radius <= 1.0
    
    if is_normalized:
        cx = center_x * config_width
        cy = center_y * config_height
        r = radius * min(config_width, config_height)  # Use smaller dimension
    else:
        cx, cy, r = center_x, center_y, radius
    
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        points.append([int(x), int(y)])
    return points


def rectangle_to_polygon(x: float, y: float, width: float, height: float,
                        config_width: int = 1920, config_height: int = 1080) -> List[List[int]]:
    """Convert rectangle to polygon (4 points).
    
    Args:
        x, y: Top-left corner (normalized 0-1 or pixel)
        width, height: Size (normalized 0-1 or pixel, -1 means full frame)
        config_width, config_height: For normalization if needed
    
    Returns:
        List of 4 polygon points in pixel coordinates
    """
    is_normalized = x <= 1.0 and y <= 1.0
    
    if is_normalized:
        x1 = x * config_width
        y1 = y * config_height
        if width == -1:
            w = config_width
        else:
            w = width * config_width
        if height == -1:
            h = config_height
        else:
            h = height * config_height
    else:
        x1, y1 = x, y
        w = width if width != -1 else config_width
        h = height if height != -1 else config_height
    
    # 4 corners: top-left, top-right, bottom-right, bottom-left
    return [
        [int(x1), int(y1)],
        [int(x1 + w), int(y1)],
        [int(x1 + w), int(y1 + h)],
        [int(x1), int(y1 + h)]
    ]


@dataclass
class CameraConfig(BaseCameraConfig):
    """Camera config with ROI rules."""
    roi_rules: List[Dict] = field(default_factory=list)
    task_name: str = ""  # Tên bài toán từ redis key, e.g. GS_HangRao, CB_GIAN_LAN

    def rules_hash(self) -> str:
        """Hash of rules for comparison."""
        return json.dumps(self.roi_rules, sort_keys=True)


class AreaMonitoringRedisSync(BaseRedisSync):
    """
    Redis sync for area_monitoring with ROI support.
    
    Extends BaseRedisSync to add:
    - ROI parsing from Redis rule data
    - on_roi_update callback
    - Normalized to pixel coordinate conversion
    """

    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_key: str,
        poll_interval: float,
        config_width: int,
        config_height: int,
        on_add: Callable[[str, str, Dict], None],
        on_remove: Callable[[str, str], None],  # (camera_id, task_name) for per-task remove
        on_restart: Callable[[str, str, Dict], None],
        on_roi_update: Callable[[str, Dict], None],
        redis_keys: Optional[List[str]] = None,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
    ):
        # Mode "ais": redis_key="ais", redis_keys=["GS_HangRao", "CB_GIAN_LAN", ...] → GET ais, use data[type_ai]
        # Legacy: redis_keys=["GS_HangRao:0", ...] → each element is a Redis key
        self._ais_mode = bool(redis_key == "ais" and redis_keys)
        if self._ais_mode:
            self._redis_keys = list(redis_keys)
            base_key = "ais"
        elif redis_keys:
            self._redis_keys = list(redis_keys)
            base_key = self._redis_keys[0]
        else:
            self._redis_keys = [redis_key] if redis_key else []
            base_key = redis_key or "GS_HangRao:0"

        self._on_add_with_config = on_add
        self._on_remove_with_config = on_remove
        self._on_restart_with_config = on_restart
        self._on_roi_update = on_roi_update

        def _on_add_wrapper(camera_id: str, rtsp_url: str) -> None:
            pass

        def _on_remove_wrapper(camera_id: str, task_name: str = "") -> None:
            on_remove(camera_id, task_name)

        def _on_restart_wrapper(camera_id: str, new_rtsp_url: str) -> None:
            pass

        super().__init__(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_key=base_key,
            poll_interval=poll_interval,
            on_add=_on_add_wrapper,
            on_remove=_on_remove_wrapper,
            on_restart=_on_restart_wrapper,
            redis_db=redis_db,
            redis_password=redis_password,
        )

        self._config_width = config_width
        self._config_height = config_height

    def _normalize_to_pixel(self, points: List[List[float]]) -> List[List[int]]:
        """Convert normalized coords (0-1) to pixel coords. Auto-detects if already pixels."""
        is_pixel = any(p[0] > 1.1 or p[1] > 1.1 for p in points)
        if is_pixel:
            return [[int(p[0]), int(p[1])] for p in points]
        return [[int(p[0] * self._config_width), int(p[1] * self._config_height)] for p in points]

    def _parse_rules(self, rules: List[Dict]) -> List[Dict]:
        """Parse Redis rules to processor format.
        
        Supports 3 types:
        - circle: {"type": "circle", "x": 0.5, "y": 0.5, "radius": 0.2}
        - rectangle: {"type": "rectangle", "x": 0, "y": 0, "width": 1, "height": 1}
        - polygon: {"type": "polygon", "points": [[...], [...]]}
        
        All types are converted to polygon format for nvdsanalytics.
        """
        out: List[Dict] = []
        for rule in rules:
            rule_type = rule.get("type", "polygon")
            rule_id = rule.get("id", len(out))
            
            polygon = None
            
            if rule_type == "circle":
                # Convert circle to polygon
                center_x = rule.get("x", 0.5)
                center_y = rule.get("y", 0.5)
                radius = rule.get("radius", 0.1)
                polygon = circle_to_polygon(
                    center_x, center_y, radius,
                    num_points=32,  # 32 points for smooth circle
                    config_width=self._config_width,
                    config_height=self._config_height
                )
                
            elif rule_type == "rectangle":
                # Convert rectangle to polygon
                x = rule.get("x", 0)
                y = rule.get("y", 0)
                width = rule.get("width", 1)
                height = rule.get("height", 1)
                polygon = rectangle_to_polygon(
                    x, y, width, height,
                    config_width=self._config_width,
                    config_height=self._config_height
                )
                
            elif rule_type == "polygon":
                # Use polygon points directly
                points = rule.get("points", [])
                if not points:
                    continue
                polygon = self._normalize_to_pixel(points)
                
            else:
                logger.warning(f"[AM_REDIS] Unknown rule type: {rule_type}, skipping")
                continue
            
            if not polygon or len(polygon) < 3:
                logger.warning(f"[AM_REDIS] Invalid polygon for rule {rule_id}, skipping")
                continue
            
            out.append({
                "roi_id": f"roi_{rule_id}",
                "polygon": polygon,
                "class_id": [0],  # Default, can be overridden
                # Note: inverse không lấy từ Redis, mà từ config.yaml
            })
        
        return out

    def _get_current(self):
        """Đọc từ Redis. Mặc định key "ais" lấy bài toán; mỗi camera chỉ lấy rtsp đầu tiên."""
        try:
            cams: Dict[str, CameraConfig] = {}
            if self._ais_mode:
                raw = self._redis.get(self._redis_key) or "{}"
                top = json.loads(raw)
                for rk in self._redis_keys:
                    data = top.get(rk, {}) or {}
                    task_name = rk
                    for cam_key in sorted(data.get("list_camera", {}).keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                        cam_data = data["list_camera"][cam_key]
                        rtsp_urls = cam_data.get("rtsp", [])
                        if not rtsp_urls:
                            continue
                        orig_id = self._parse_camera_id(cam_data, cam_key)
                        roi_rules = self._parse_rules(cam_data.get("rule", []))
                        composite = f"{rk}__{cam_key}"
                        camera_id = f"{orig_id}" if len(self._redis_keys) > 1 else orig_id
                        # Chỉ lấy rtsp đầu tiên cho mỗi camera
                        cams[composite] = CameraConfig(
                            camera_id=camera_id,
                            rtsp_url=rtsp_urls[0],
                            roi_rules=roi_rules,
                            task_name=task_name,
                        )
                return cams
            for rk in self._redis_keys:
                raw = self._redis.get(rk) or "{}"
                data = json.loads(raw)
                task_name = (rk.split(":")[0] if ":" in rk else rk).strip()
                for cam_key in sorted(data.get("list_camera", {}).keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                    cam_data = data["list_camera"][cam_key]
                    rtsp_urls = cam_data.get("rtsp", [])
                    if not rtsp_urls:
                        continue
                    orig_id = self._parse_camera_id(cam_data, cam_key)
                    roi_rules = self._parse_rules(cam_data.get("rule", []))
                    composite = f"{rk}__{cam_key}"
                    camera_id = f"{orig_id}" if len(self._redis_keys) > 1 else orig_id
                    # Chỉ lấy rtsp đầu tiên cho mỗi camera
                    cams[composite] = CameraConfig(
                        camera_id=camera_id,
                        rtsp_url=rtsp_urls[0],
                        roi_rules=roi_rules,
                        task_name=task_name,
                    )
            return cams
        except Exception as e:
            logger.error("[AM_REDIS] Error fetching cameras: %s", e, exc_info=True)
            return None

    def _camera_config_dict(self, cfg: CameraConfig) -> Dict:
        """Convert CameraConfig to dict with ROI filtering và task_name."""
        d = {
            "config_width": self._config_width,
            "config_height": self._config_height,
            "roi_filtering": cfg.roi_rules,
        }
        if getattr(cfg, "task_name", ""):
            d["task_name"] = cfg.task_name
        return d

    def _sync_once(self) -> None:
        """Override to add ROI update detection and camera_config to callbacks."""
        curr = self._get_current()
        if curr is None:
            return

        prev_keys = set(self._prev.keys())
        curr_keys = set(curr.keys())

        def _sort_key(x: str):
            if "__" in x and str(x).split("__")[-1].isdigit():
                return (0, int(x.split("__")[-1]))
            return (1, x)

        # Remove (process in sorted order for consistency). Pass (camera_id, task_name) for per-task remove.
        for key in sorted(prev_keys - curr_keys, key=_sort_key):
            old = self._prev[key]
            try:
                task_name = getattr(old, "task_name", "") or ""
                self._on_remove_with_config(old.camera_id, task_name)
                logger.info("[AM_REDIS][REMOVE] %s task=%s", old.camera_id, task_name)
            except Exception as e:
                logger.error("[AM_REDIS] on_remove failed for %s: %s", old.camera_id, e, exc_info=True)

        # Add (process in sorted order)
        add_keys = sorted(curr_keys - prev_keys, key=_sort_key)
        for idx, key in enumerate(add_keys):
            cfg = curr[key]
            try:
                # Add delay between multiple camera additions (except first one)
                if idx > 0:
                    import time
                    time.sleep(2.5)  # Wait 2.5s between camera additions
                camera_config = self._camera_config_dict(cfg)
                self._on_add_with_config(cfg.camera_id, cfg.rtsp_url, camera_config)
                logger.info("[AM_REDIS][ADD] %s (%d ROI)", cfg.camera_id, len(cfg.roi_rules))
            except Exception as e:
                logger.error("[AM_REDIS] on_add failed for %s: %s", cfg.camera_id, e, exc_info=True)

        # Update existing (process in sorted order)
        for key in sorted(curr_keys & prev_keys, key=_sort_key):
            new = curr[key]
            old = self._prev[key]

            if new.rtsp_url != old.rtsp_url:
                try:
                    camera_config = self._camera_config_dict(new)
                    self._on_restart_with_config(new.camera_id, new.rtsp_url, camera_config)
                    logger.info("[AM_REDIS][UPDATE_URI] %s", new.camera_id)
                except Exception as e:
                    logger.error("[AM_REDIS] on_restart failed for %s: %s", new.camera_id, e, exc_info=True)
            elif isinstance(new, CameraConfig) and isinstance(old, CameraConfig):
                if new.rules_hash() != old.rules_hash():
                    try:
                        camera_config = self._camera_config_dict(new)
                        self._on_roi_update(new.camera_id, camera_config)
                        logger.info("[AM_REDIS][UPDATE_ROI] %s (%d ROI)", new.camera_id, len(new.roi_rules))
                    except Exception as e:
                        logger.error("[AM_REDIS] on_roi_update failed for %s: %s", new.camera_id, e, exc_info=True)

        self._prev = curr