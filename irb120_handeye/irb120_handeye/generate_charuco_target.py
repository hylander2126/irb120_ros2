#!/usr/bin/env python3
"""Generate a print-ready ChArUco hand-eye calibration target.

Replaces the ArUco grid board (irb_target_image.png / irb_ChArUco.xml) with
a real ChArUco board: checkerboard-corner pose estimation is more accurate
and more occlusion-tolerant than the marker-corners-only pose used by the
ArUco grid board / the MoveIt HandEyeCalibration panel.

Run once whenever BOARD_* below changes:

  ros2 run irb120_handeye generate_charuco_target

Writes, to the home directory:
  - charuco_target.pdf  — the print-ready page. The PDF page size is set
    to the exact physical page dimensions (e.g. 8.5x11in for letter), so
    print it at 100% / "actual size" (do NOT let the print dialog "fit to
    page" or "shrink to fit") and it will come out to scale. A plain PNG
    carries no reliable page-size information, which is why PDF readers/
    printers asked to convert one to PDF often get the scale wrong even
    at "100%" -- this script writes the PDF directly instead to avoid
    that step.
  - charuco_target.png  — same page, as a raster image for quick preview;
    do not print from this file (see above).
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

DICTIONARY_NAME = 'DICT_5X5_250'  # matches the ArUco grid board already in use, no reason to change
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_LENGTH_M = 0.032
MARKER_LENGTH_M = 0.024

# Page sizes given as (width, height) in inches -- both entries are already
# portrait (width < height); that's the default layout.
PAGE_SIZES_IN = {'letter': (8.5, 11.0), 'a4': (8.27, 11.69)}
DEFAULT_PAGE = 'letter'
DEFAULT_DPI = 300

# Physical backing board the printed page gets cut down and mounted to. The
# cut line drawn on the page is exactly this size (not smaller) -- the
# ChArUco board is simply centered inside it, whatever margin that leaves.
BACKING_BOARD_W_MM = 240.0
BACKING_BOARD_H_MM = 185.0


def _default_out_dir() -> str:
    return os.path.expanduser('~')


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
    out_dir = out_dir or _default_out_dir()

    board = build_board()

    px_per_m = dpi / 0.0254
    board_w_px = int(round(SQUARES_X * SQUARE_LENGTH_M * px_per_m))
    board_h_px = int(round(SQUARES_Y * SQUARE_LENGTH_M * px_per_m))
    board_img = board.draw((board_w_px, board_h_px))

    page_w_in, page_h_in = PAGE_SIZES_IN[page]
    page_w_px, page_h_px = int(round(page_w_in * dpi)), int(round(page_h_in * dpi))

    # Cut line = the backing board's exact physical size, laid out to match
    # the page's portrait orientation (the board's shorter side maps to the
    # page's width, its longer side to the page's height) regardless of
    # which of BACKING_BOARD_W_MM/H_MM happens to be larger.
    cut_w_mm = min(BACKING_BOARD_W_MM, BACKING_BOARD_H_MM)
    cut_h_mm = max(BACKING_BOARD_W_MM, BACKING_BOARD_H_MM)
    cut_w_px = int(round(cut_w_mm / 1000 * px_per_m))
    cut_h_px = int(round(cut_h_mm / 1000 * px_per_m))

    # The board itself may need to be rotated 90 degrees to match the
    # portrait-shaped cut box -- pick whichever orientation fits with the
    # most margin to spare (a physical rotation of the printed board has
    # no effect on ChArUco detection or calibration).
    orientations = [(board_w_px, board_h_px, board_img)]
    if board_w_px != board_h_px:
        orientations.append((board_h_px, board_w_px, cv2.rotate(board_img, cv2.ROTATE_90_CLOCKWISE)))
    fitting = [(w, h, img) for w, h, img in orientations if w <= cut_w_px and h <= cut_h_px]
    if not fitting:
        raise ValueError(
            f'{SQUARES_X}x{SQUARES_Y} squares @ {SQUARE_LENGTH_M * 1000:.1f}mm does not fit '
            f'the {BACKING_BOARD_W_MM:.0f}x{BACKING_BOARD_H_MM:.0f}mm backing board in either '
            'orientation -- shrink the board or use a bigger backing board.')
    board_w_px, board_h_px, board_img = max(fitting, key=lambda t: min(cut_w_px - t[0], cut_h_px - t[1]))

    if cut_w_px > page_w_px or cut_h_px > page_h_px:
        raise ValueError(
            f'Backing board {cut_w_mm:.1f}x{cut_h_mm:.1f}mm does not fit on {page} '
            f'({page_w_in}x{page_h_in}in) at {dpi} DPI -- pick a bigger page.')

    # Lay out, top to bottom: the "cut here" label, the cut box (board
    # centered inside it), a gap, then the scale bar + its two text lines.
    # The whole stack is centered vertically (and the cut box horizontally)
    # on the page.
    label_h_px = 40
    scale_gap_px = int(round(0.3 * dpi))
    scale_block_h_px = int(round(0.9 * dpi))
    stack_h_px = label_h_px + cut_h_px + scale_gap_px + scale_block_h_px
    stack_y0 = max((page_h_px - stack_h_px) // 2, 0)

    cut_x0 = (page_w_px - cut_w_px) // 2
    cut_y0 = stack_y0 + label_h_px
    cut_x1, cut_y1 = cut_x0 + cut_w_px, cut_y0 + cut_h_px

    x0 = cut_x0 + (cut_w_px - board_w_px) // 2
    y0 = cut_y0 + (cut_h_px - board_h_px) // 2

    page_img = np.full((page_h_px, page_w_px), 255, dtype=np.uint8)
    page_img[y0:y0 + board_h_px, x0:x0 + board_w_px] = board_img

    # Dashed cut line: trim to this box (the backing board's exact size)
    # before mounting, with the board centered inside it.
    dash, gap = 12, 8
    for xa, ya, xb, yb in ((cut_x0, cut_y0, cut_x1, cut_y0), (cut_x0, cut_y1, cut_x1, cut_y1)):
        for x in range(xa, xb, dash + gap):
            cv2.line(page_img, (x, ya), (min(x + dash, xb), yb), 0, 2)
    for xa, ya, xb, yb in ((cut_x0, cut_y0, cut_x0, cut_y1), (cut_x1, cut_y0, cut_x1, cut_y1)):
        for y in range(ya, yb, dash + gap):
            cv2.line(page_img, (xa, y), (xb, min(y + dash, yb)), 0, 2)
    cv2.putText(page_img,
                f'cut here -- {cut_w_mm:.0f}x{cut_h_mm:.0f}mm backing board',
                (cut_x0, max(cut_y0 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)

    # Scale bar + label burned into the page itself, so print scale is
    # checkable with a ruler alone and "print at 100%" can't be missed.
    bar_len_px = int(round(0.050 * px_per_m))  # 50mm reference
    bar_y = cut_y1 + scale_gap_px
    bar_x = cut_x0
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

    # Write the PDF directly at the exact physical page size (inches ->
    # points is exact and unambiguous) rather than relying on a PNG-to-PDF
    # conversion, which has no reliable way to know the image's intended
    # physical size and commonly rescales it even when "100%" is selected.
    pdf_path = os.path.join(out_dir, 'charuco_target.pdf')
    fig = plt.figure(figsize=(page_w_in, page_h_in), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(page_img, cmap='gray', vmin=0, vmax=255, interpolation='none')
    ax.axis('off')
    fig.savefig(pdf_path, dpi=dpi)
    plt.close(fig)

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

    print(f'Wrote {pdf_path}  ({page_w_in}x{page_h_in}in {page} page @ {dpi} DPI) -- print this one')
    print(f'Wrote {img_path}  (preview only, do not print)')
    print(f'Wrote {yaml_path}')
    print('Print the PDF at 100% / actual size (disable "fit to page"/"shrink to fit"), then '
          'verify the 50mm reference bar with a ruler before mounting or calibrating.')
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--page', default=DEFAULT_PAGE, choices=sorted(PAGE_SIZES_IN),
                        help='Page size to lay the board out on (default: letter).')
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI)
    parser.add_argument('--out-dir', default=None,
                        help='Defaults to the home directory.')
    args = parser.parse_args()
    generate(page=args.page, dpi=args.dpi, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
