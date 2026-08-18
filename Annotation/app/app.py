from __future__ import annotations

import argparse
import warnings


def _suppress_noisy_dependency_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*HTTP_422_UNPROCESSABLE_CONTENT.*",
        category=Warning,
    )


_suppress_noisy_dependency_warnings()

from target_video_search.ui import build_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the video target search UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
