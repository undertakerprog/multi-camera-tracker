import argparse
import json
import logging
from pathlib import Path

from src.app import TargetTrackingApp
from src.cameras.basler_camera import BaslerCameraSource
from src.common import StageProfiler
from src.ui import OpenCVDisplay


LOG = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_tracking_config(config: dict, profile_name: str) -> dict:
    """Merge the base tracking block with the selected profile overrides."""
    base = dict(config.get("tracking", {}))
    profiles = config.get("profiles", {})
    overrides = profiles.get(profile_name)
    if overrides is None:
        if profiles:
            LOG.warning("Unknown profile '%s'; using base tracking config", profile_name)
        return base
    base.update(overrides)
    return base


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(value) for value in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be 'x,y,w,h'")
    return parts[0], parts[1], parts[2], parts[3]


def build_parser(config: dict, profile_name: str, parents: list) -> argparse.ArgumentParser:
    camera_config = config.get("camera", {})
    tracking_config = resolve_tracking_config(config, profile_name)
    display_config = config.get("display", {})

    parser = argparse.ArgumentParser(
        description="Jetson target tracking prototype",
        parents=parents,
    )
    parser.add_argument("--tracker", default=tracking_config.get("tracker", "CSRT"),
                        choices=["CSRT", "KCF", "MOSSE", "MIL"])
    parser.add_argument("--tracking-scale", type=float, default=tracking_config.get("scale", 0.75))
    parser.add_argument("--no-roi", action="store_true",
                        default=not tracking_config.get("use_roi", True))
    parser.add_argument("--roi-scale", type=float, default=tracking_config.get("roi_scale", 2.5))
    parser.add_argument("--roi-edge-margin", type=float,
                        default=tracking_config.get("roi_edge_margin", 0.15))
    parser.add_argument("--roi-min-window", type=int,
                        default=tracking_config.get("roi_min_window", 64))
    parser.add_argument("--smooth-alpha", type=float,
                        default=tracking_config.get("smooth_alpha", 0.35))
    parser.add_argument("--max-lost-frames", type=int,
                        default=tracking_config.get("max_lost_frames", 120))
    parser.add_argument("--disable-reacquire", action="store_true",
                        default=not tracking_config.get("reacquire_enabled", True))
    parser.add_argument("--reacquire-score", type=float,
                        default=tracking_config.get("reacquire_score", 0.62))
    parser.add_argument("--reacquire-search", type=float,
                        default=tracking_config.get("reacquire_search", 3.0))
    parser.add_argument("--reacquire-cooldown", type=int,
                        default=tracking_config.get("reacquire_cooldown", 8))
    parser.add_argument("--disable-global-reacquire", action="store_true",
                        default=not tracking_config.get("global_reacquire_enabled", True))
    parser.add_argument("--global-reacquire-after", type=int,
                        default=tracking_config.get("global_reacquire_after", 8))
    parser.add_argument("--global-reacquire-interval", type=int,
                        default=tracking_config.get("global_reacquire_interval", 5))
    parser.add_argument("--global-reacquire-score", type=float,
                        default=tracking_config.get("global_reacquire_score", 0.72))
    parser.add_argument("--global-reacquire-scale", type=float,
                        default=tracking_config.get("global_reacquire_scale", 0.5))
    parser.add_argument("--template-update-interval", type=int,
                        default=tracking_config.get("template_update_interval", 15))
    parser.add_argument("--template-min-confirmations", type=int,
                        default=tracking_config.get("template_min_confirmations", 5))
    parser.add_argument("--verify-interval", type=int,
                        default=tracking_config.get("verify_interval", 20))
    parser.add_argument("--verify-min-score", type=float,
                        default=tracking_config.get("verify_min_score", 0.45))

    parser.add_argument("--serial", default=camera_config.get("serial"))
    parser.add_argument("--exposure-us", type=int, default=camera_config.get("exposure_us"))
    parser.add_argument("--gain-db", type=float, default=camera_config.get("gain_db"))
    parser.add_argument("--exposure-auto", default=camera_config.get("exposure_auto", "Off"),
                        choices=["Off", "Once", "Continuous"])
    parser.add_argument("--gain-auto", default=camera_config.get("gain_auto", "Off"),
                        choices=["Off", "Once", "Continuous"])
    parser.add_argument("--pixel-format", default=camera_config.get("pixel_format", "Mono8"))
    parser.add_argument("--camera-width", type=int, default=camera_config.get("width"))
    parser.add_argument("--camera-height", type=int, default=camera_config.get("height"))
    parser.add_argument("--offset-x", type=int, default=camera_config.get("offset_x"))
    parser.add_argument("--offset-y", type=int, default=camera_config.get("offset_y"))
    parser.add_argument("--no-center-roi", action="store_true",
                        default=not camera_config.get("center_roi", True))
    parser.add_argument("--camera-fps", type=float, default=camera_config.get("frame_rate"))
    parser.add_argument("--display-auto-contrast", action="store_true",
                        default=display_config.get("auto_contrast", False))

    parser.add_argument("--benchmark", action="store_true",
                        help="Run the headless capture/tracking benchmark and exit")
    parser.add_argument("--benchmark-roi", type=str, default=None,
                        help="ROI 'x,y,w,h' for the tracking benchmark phase")
    parser.add_argument("--no-profiler", action="store_true",
                        help="Disable per-stage timing telemetry")
    return parser


def build_camera(args) -> BaslerCameraSource:
    return BaslerCameraSource(
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
        convert_to_bgr=False,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="config.json")
    bootstrap.add_argument("--profile", default=None,
                           help="Tracking profile: quality | balanced | fast")
    known_args, _ = bootstrap.parse_known_args()

    config = load_config(known_args.config)
    profile_name = known_args.profile or config.get("profile", "balanced")

    parser = build_parser(config, profile_name, parents=[bootstrap])
    args = parser.parse_args()
    tracking_config = resolve_tracking_config(config, profile_name)

    camera = build_camera(args)

    if args.benchmark:
        from src.benchmark import format_report, run_benchmark

        tracking_config["use_roi"] = not args.no_roi
        tracking_config["roi_scale"] = args.roi_scale
        roi = parse_roi(args.benchmark_roi)
        LOG.info("Running benchmark (profile=%s, roi=%s)", profile_name, roi)
        results = run_benchmark(camera, tracking_config, roi=roi)
        print(format_report(results))
        return

    display = OpenCVDisplay(auto_contrast=args.display_auto_contrast)
    profiler = StageProfiler(enabled=not args.no_profiler)
    app = TargetTrackingApp(
        camera_source=camera,
        tracker_type=args.tracker,
        tracking_scale=args.tracking_scale,
        use_roi=not args.no_roi,
        roi_scale=args.roi_scale,
        roi_edge_margin=args.roi_edge_margin,
        roi_min_window=args.roi_min_window,
        smoothing_alpha=args.smooth_alpha,
        max_lost_frames=args.max_lost_frames,
        prediction_display_frames=tracking_config.get("prediction_display_frames", 15),
        reacquire_enabled=not args.disable_reacquire,
        reacquire_min_score=args.reacquire_score,
        reacquire_search_expansion=args.reacquire_search,
        reacquire_cooldown=args.reacquire_cooldown,
        global_reacquire_enabled=not args.disable_global_reacquire,
        global_reacquire_after=args.global_reacquire_after,
        global_reacquire_interval=args.global_reacquire_interval,
        global_reacquire_score=args.global_reacquire_score,
        global_reacquire_scale=args.global_reacquire_scale,
        template_update_interval=args.template_update_interval,
        template_min_confirmations=args.template_min_confirmations,
        verify_interval=args.verify_interval,
        verify_min_score=args.verify_min_score,
        profiler=profiler,
        display=display,
    )
    app.run()


if __name__ == "__main__":
    main()
