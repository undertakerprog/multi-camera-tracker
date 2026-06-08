import argparse

from src.app import TargetTrackingApp
from src.cameras.basler_camera import BaslerCameraSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson target tracking prototype")
    parser.add_argument(
        "--tracker",
        default="CSRT",
        choices=["CSRT", "KCF", "MOSSE", "MIL"],
        help="OpenCV tracker type",
    )
    parser.add_argument("--serial", default=None, help="Basler camera serial number")
    parser.add_argument("--exposure-us", type=int, default=None)
    parser.add_argument("--gain-db", type=float, default=None)
    parser.add_argument(
        "--pixel-format",
        default="Mono8",
        help="Basler PixelFormat value, e.g. Mono8",
    )
    args = parser.parse_args()

    camera = BaslerCameraSource(
        serial_number=args.serial,
        exposure_us=args.exposure_us,
        gain_db=args.gain_db,
        pixel_format=args.pixel_format,
    )
    app = TargetTrackingApp(camera_source=camera, tracker_type=args.tracker)
    app.run()


if __name__ == "__main__":
    main()
