import argparse
import json
from pathlib import Path

from src.app import TargetTrackingApp
from src.cameras.basler_camera import BaslerCameraSource
from src.ui import OpenCVDisplay


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="config.json")
    known_args, _ = config_parser.parse_known_args()

    config = load_config(known_args.config)
    camera_config = config.get("camera", {})
    tracking_config = config.get("tracking", {})
    display_config = config.get("display", {})

    parser = argparse.ArgumentParser(
        description="Jetson target tracking prototype",
        parents=[config_parser],
    )
    parser.add_argument(
        "--tracker",
        default=tracking_config.get("tracker", "CSRT"),
        choices=["CSRT", "KCF", "MOSSE", "MIL"],
    )
    parser.add_argument(
        "--tracking-scale",
        type=float,
        default=tracking_config.get("scale", 0.75),
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=tracking_config.get("smooth_alpha", 0.35),
    )
    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=tracking_config.get("max_lost_frames", 120),
    )
    parser.add_argument(
        "--disable-reacquire",
        action="store_true",
        default=not tracking_config.get("reacquire_enabled", True),
    )
    parser.add_argument(
        "--reacquire-score",
        type=float,
        default=tracking_config.get("reacquire_score", 0.62),
    )
    parser.add_argument(
        "--reacquire-search",
        type=float,
        default=tracking_config.get("reacquire_search", 3.0),
    )
    parser.add_argument(
        "--disable-global-reacquire",
        action="store_true",
        default=not tracking_config.get("global_reacquire_enabled", True),
    )
    parser.add_argument(
        "--global-reacquire-after",
        type=int,
        default=tracking_config.get("global_reacquire_after", 8),
    )
    parser.add_argument(
        "--global-reacquire-interval",
        type=int,
        default=tracking_config.get("global_reacquire_interval", 5),
    )
    parser.add_argument(
        "--global-reacquire-score",
        type=float,
        default=tracking_config.get("global_reacquire_score", 0.72),
    )
    parser.add_argument(
        "--global-reacquire-scale",
        type=float,
        default=tracking_config.get("global_reacquire_scale", 0.5),
    )
    parser.add_argument(
        "--template-update-interval",
        type=int,
        default=tracking_config.get("template_update_interval", 15),
    )
    parser.add_argument("--serial", default=camera_config.get("serial"))
    parser.add_argument(
        "--exposure-us",
        type=int,
        default=camera_config.get("exposure_us"),
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        default=camera_config.get("gain_db"),
    )
    parser.add_argument(
        "--exposure-auto",
        default=camera_config.get("exposure_auto", "Off"),
        choices=["Off", "Once", "Continuous"],
    )
    parser.add_argument(
        "--gain-auto",
        default=camera_config.get("gain_auto", "Off"),
        choices=["Off", "Once", "Continuous"],
    )
    parser.add_argument(
        "--pixel-format",
        default=camera_config.get("pixel_format", "Mono8"),
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=camera_config.get("width"),
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=camera_config.get("height"),
    )
    parser.add_argument("--offset-x", type=int, default=camera_config.get("offset_x"))
    parser.add_argument("--offset-y", type=int, default=camera_config.get("offset_y"))
    parser.add_argument(
        "--no-center-roi",
        action="store_true",
        default=not camera_config.get("center_roi", True),
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=camera_config.get("frame_rate"),
    )
    parser.add_argument(
        "--display-auto-contrast",
        action="store_true",
        default=display_config.get("auto_contrast", False),
    )
    args = parser.parse_args()

    camera = BaslerCameraSource(
        serial_number=args.serial,
        exposure_us=args.exposure_us,
        gain_db=args.gain_db,
        exposure_auto=args.exposure_auto,
        gain_auto=args.gain_auto,
        pixel_format=args.pixel_format,
        width=args.camera_width,
        height=args.camera_height,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        center_roi=not args.no_center_roi,
        frame_rate=args.camera_fps,
    )
    display = OpenCVDisplay(auto_contrast=args.display_auto_contrast)
    app = TargetTrackingApp(
        camera_source=camera,
        tracker_type=args.tracker,
        tracking_scale=args.tracking_scale,
        smoothing_alpha=args.smooth_alpha,
        max_lost_frames=args.max_lost_frames,
        reacquire_enabled=not args.disable_reacquire,
        reacquire_min_score=args.reacquire_score,
        reacquire_search_expansion=args.reacquire_search,
        global_reacquire_enabled=not args.disable_global_reacquire,
        global_reacquire_after=args.global_reacquire_after,
        global_reacquire_interval=args.global_reacquire_interval,
        global_reacquire_score=args.global_reacquire_score,
        global_reacquire_scale=args.global_reacquire_scale,
        template_update_interval=args.template_update_interval,
        display=display,
    )
    app.run()


if __name__ == "__main__":
    main()
