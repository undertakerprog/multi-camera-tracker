import argparse

from src.app import TargetTrackingApp
from src.cameras.ximea_camera import XimeaCameraSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson target tracking prototype")
    parser.add_argument(
        "--tracker",
        default="CSRT",
        choices=["CSRT", "KCF", "MOSSE", "MIL"],
        help="OpenCV tracker type",
    )
    parser.add_argument("--exposure-us", type=int, default=None)
    parser.add_argument("--gain-db", type=float, default=None)
    args = parser.parse_args()

    camera = XimeaCameraSource(exposure_us=args.exposure_us, gain_db=args.gain_db)
    app = TargetTrackingApp(camera_source=camera, tracker_type=args.tracker)
    app.run()


if __name__ == "__main__":
    main()
