#!/usr/bin/env python3
"""Generate a print-ready ChArUco hand-eye calibration target.

Replaces the ArUco grid board (irb_target_image.png / irb_ChArUco.xml) with
a real ChArUco board: checkerboard-corner pose estimation is more accurate
and more occlusion-tolerant than the marker-corners-only pose used by the
ArUco grid board / the MoveIt HandEyeCalibration panel.

Run once whenever BOARD_* below changes:

  ros2 run irb120_handeye generate_charuco_target

Writes, under calibrations/:
  - charuco_target.png  — the print-ready page (print at 100% / "actual
    size"; do NOT let the print dialog "fit to page", or the board's real
    square size will no longer match square_length_m below and every
    downstream calibration will carry a systematic scale error).
  - charuco_board.yaml  — the board geometry, read back in by
    run_handeye_calibration.py so detection always matches what was
    printed. Single source of truth: don't hand-edit the numbers in one
    file without regenerating the other.

After printing, mount on a rigid flat backing (not foam-core) and verify
the burned-in 50mm reference bar with a ruler/calipers before trusting the
nominal square_length_m — a printer's own scaling error is otherwise
invisible until it shows up as calibration bias.
"""

import argparse
import os

import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

DICTIONARY_NAME = 'DICT_5X5_250'  # matches the ArUco grid board already in use, no reason to change
SQUARES_X, SQUARES_Y = 7, 5
SQUARE_LENGTH_M = 0.025
MARKER_LENGTH_M = 0.018

PAGE_SIZES_IN = {'letter': (8.5, 11.0), 'a4': (8.27, 11.69)}
DEFAULT_PAGE = 'letter'
DEFAULT_DPI = 300
TOP_MARGIN_IN = 0.5


def _calibrations_dir() -> str:
    # Prefer the source tree so the result lands somewhere git-tracked and
    # visible in an editor; fall back to the installed share dir if this is
    # ever run from an installed-only environment.
    here = os.path.dirname(os.path.abspath(__file__))
    src_calib = os.path.normpath(os.path.join(here, '..', 'calibrations'))
    if os.path.isdir(src_calib):
        return src_calib
    return os.path.join(get_package_share_directory('irb120_handeye'), 'calibrations')


def build_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY_NAME))
    # CharucoBoard_create (legacy factory), not the newer CharucoBoard(...)
    # constructor: on this OpenCV build (4.6.0) the newer constructor's
    # .draw() segfaults. CharucoBoard_create's .draw() does not.
    return cv2.aruco.CharucoBoard_create(
        SQUARES_X, SQUARES_Y, SQUARE_LENGTH_M, MARKER_LENGTH_M, dictionary)


def generate(page: str = DEFAULT_PAGE, dpi: int = DEFAULT_DPI, out_dir: str = None) -> str:
    if page not in PAGE_SIZES_IN:
        raise ValueError(f'Unknown page size {page!r}; choose from {sorted(PAGE_SIZES_IN)}')
    out_dir = out_dir or _calibrations_dir()

    board = build_board()

    px_per_m = dpi / 0.0254
    board_w_px = int(round(SQUARES_X * SQUARE_LENGTH_M * px_per_m))
    board_h_px = int(round(SQUARES_Y * SQUARE_LENGTH_M * px_per_m))
    board_img = board.draw((board_w_px, board_h_px))

    page_w_in, page_h_in = PAGE_SIZES_IN[page]
    page_w_px, page_h_px = int(round(page_w_in * dpi)), int(round(page_h_in * dpi))
    if board_w_px > page_w_px or board_h_px > page_h_px:
        raise ValueError(
            f'{SQUARES_X}x{SQUARES_Y} squares @ {SQUARE_LENGTH_M * 1000:.1f}mm is '
            f'{board_w_px / dpi:.2f}x{board_h_px / dpi:.2f}in, too big for {page} '
            f'({page_w_in}x{page_h_in}in) at {dpi} DPI — shrink the board or pick a bigger page.')

    page_img = np.full((page_h_px, page_w_px), 255, dtype=np.uint8)
    x0 = (page_w_px - board_w_px) // 2
    y0 = int(round(TOP_MARGIN_IN * dpi))
    page_img[y0:y0 + board_h_px, x0:x0 + board_w_px] = board_img

    # Scale bar + label burned into the page itself, so print scale is
    # checkable with a ruler alone and "print at 100%" can't be missed.
    bar_len_px = int(round(0.050 * px_per_m))  # 50mm reference
    bar_y = min(y0 + board_h_px + int(round(0.3 * dpi)), page_h_px - int(round(0.6 * dpi)))
    bar_x = x0
    cv2.line(page_img, (bar_x, bar_y), (bar_x + bar_len_px, bar_y), 0, 3)
    for tick_x in (bar_x, bar_x + bar_len_px):
        cv2.line(page_img, (tick_x, bar_y - 10), (tick_x, bar_y + 10), 0, 3)
    cv2.putText(page_img, '50.0 mm reference -- verify with a ruler/calipers after printing',
                (bar_x, bar_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2, cv2.LINE_AA)
    cv2.putText(page_img,
                f'ChArUco {SQUARES_X}x{SQUARES_Y}, square={SQUARE_LENGTH_M * 1000:.1f}mm, '
                f'marker={MARKER_LENGTH_M * 1000:.1f}mm, dict={DICTIONARY_NAME} -- '
                'PRINT AT 100% / ACTUAL SIZE, DO NOT "FIT TO PAGE"',
                (bar_x, bar_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)

    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, 'charuco_target.png')
    cv2.imwrite(img_path, page_img)

    yaml_path = os.path.join(out_dir, 'charuco_board.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            'dictionary': DICTIONARY_NAME,
            'squares_x': SQUARES_X,
            'squares_y': SQUARES_Y,
            'square_length_m': SQUARE_LENGTH_M,
            'marker_length_m': MARKER_LENGTH_M,
            'note': ('square_length_m/marker_length_m are the DESIGN values used to render '
                     'charuco_target.png. After printing and mounting, measure the actual '
                     'square size with calipers; if it differs, pass --square-length-m to '
                     'run_handeye_calibration rather than editing this file.'),
        }, f, sort_keys=False)

    print(f'Wrote {img_path}  ({page_w_in}x{page_h_in}in {page} page @ {dpi} DPI)')
    print(f'Wrote {yaml_path}')
    print('Print at 100% / actual size (disable "fit to page"/"shrink to fit"), then verify '
          'the 50mm reference bar with a ruler before mounting or calibrating.')
    return img_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--page', default=DEFAULT_PAGE, choices=sorted(PAGE_SIZES_IN),
                        help='Page size to lay the board out on (default: letter).')
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI)
    parser.add_argument('--out-dir', default=None,
                        help='Defaults to the calibrations/ dir next to this package.')
    args = parser.parse_args()
    generate(page=args.page, dpi=args.dpi, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
