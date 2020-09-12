# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:04:41 2020

@author: ASUS
"""

import os
import cv2
import numpy as np
import solver
import cv_helper as cv
from digit_recognizer import DigitRecogniser


def inner_sudoku_bbox(img, bbox):
    """
    Alters a bounding box so that it captures the inside of a grid, ignoring the outline. The width of the outline
    is estimated from the image itself. Somewhat specific to Sudoku grids.
    """

    # Estimate the average width of the outline by looking at the middle of the grid where we expect to encounter
    # a number of  horizontal lines that should be of equal with to the outline.
    line_widths = []
    counter = 0
    midpoint = int(bbox[0][0] + (bbox[1][0] - bbox[0][0]) / 2)
    top = int(min([bbox[0][1], bbox[1][1]]))
    bottom = int(max([bbox[2][1], bbox[3][1]]))
    for x in range(midpoint - 5, midpoint + 5):
        for y in range(top, bottom):
            if img.item(y, x) == 255:  # If it's a white pixel start counting
                counter += 1
            else:  # Otherwise stop counting and include the reading if it's valid
                if counter > 0:
                    line_widths.append(counter)
                counter = 0

    avg_width = int(np.mean(line_widths))

    # Modify the bounding box accordingly
    bbox[0][0] += avg_width  # Top left
    bbox[0][1] += avg_width

    bbox[1][0] -= avg_width  # Top right
    bbox[1][1] += avg_width

    bbox[2][0] -= avg_width  # Bottom right
    bbox[2][1] -= avg_width

    bbox[3][0] += avg_width  # Bottom left
    bbox[3][1] -= avg_width

    return bbox


def pre_process_image(img, skip_dilate=False):
    
    """Blurs, applies adaptive thresholding, inverts colour and dilates the image."""
    # Blur to improve the efficacy of the thresholding and minimise the number of contours
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    
    # Block size of 5 gives quite a harsh thresholding, but this is required to extract the outer grid from
    # the inner grid from some photos
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)
    proc = cv2.bitwise_not(proc, proc)
    cv2.imshow('',proc)
    
    # Dilate the image to increase the size of the grid lines. Should increase reliability in finding the grids

    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc


def crop_grid(img):
    """Identify the Sudoku grid, crop it from the image and skew it into a square to compensate for the camera angle."""
    proc = pre_process_image(img)

    # Gets the four corners of the inner contours of the image. Compensates for the grid outline accurately.
    bbox, area, outer, outer_area = cv.find_four_corners_inner(proc)

    # If the inner contours are less than 95% of the outer contour we may be losing valuable information.
    # Image 8 is one such example of this occurring
    # In these cases, use a different algorithm to estimate the width of the Sudoku border
    if (area / outer_area) < 0.95:
        bbox = cv.get_four_corners(outer)
        bbox = inner_sudoku_bbox(proc, bbox)  # Estimates the width of the grid outline
        area = outer_area

    # Some images have the Sudoku grid surrounded by a large rectangular box and the contour algorithm will pick that
    # instead of the Sudoku grid. In these cases, using the largest feature algorithm is more reliable. Identify when
    # the contour area meets a threshold ratio of the image size.
    if (area / (proc.shape[0] * proc.shape[1])) > 0.95:
        # Find the largest connected pixel structure, this should be the outer grid for Sudoku
        proc, bbox, seed = cv.find_largest_feature(proc)
        bbox = inner_sudoku_bbox(proc, bbox)  # Estimates the width of the grid outline

    # Return a warped version of the image
    return cv.crop_and_warp(img, bbox, square=True)



def infer_grid(img, border=15):
    """
    Use Hough lines, linear algebra and estimate based on the face that a Sudoku is a square grid of 81 squares to infer
    a set of points that defined the corners of each box in the Sudoku grid.
    """

    # Based on the three test images used during development, the pre-processing below is the most reliable
    # Still can be optimised greatly and made to work in more general cases

    # Having a border around the image helps the infer_grid function find the points
    img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, None, (0, 0, 0))

    pproc = cv2.GaussianBlur(img, (3, 3), 0)
    pproc = cv2.adaptiveThreshold(pproc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    pproc = cv2.bitwise_not(pproc, pproc)

    # Get all possible Hough lines and then merge them down
    lines = cv2.HoughLines(pproc, 1, np.pi / 180, 200)

    height, width = img.shape[:2]
    line_threshold = int(np.mean([height, width]) / 30)
    lines = cv.merge_related_lines(pproc, lines, line_threshold)  # Dynamic threshold for merging lines
    lines = cv.filter_related_lines(lines, line_threshold)  # Filter lines that are similar to each other or off axis
    points = cv.grid_intersections(lines)  # Get all grid intersection points from the lines
    points, grid = extract_sudoku_grid(img, points, border)  # From the points we have, infer the remaining points

    # Fail if we don't have all the grid points by now
    if len(points) != 100:
        raise AssertionError('Could not extract all 100 grid points from the image.')

    rects = cv.grid_to_bbox(grid)
    return img, points, rects


def extract_sudoku_grid(img, points, border=15):
    """Uses the points found from the preprocessing steps and attempts to fill in any gaps."""

    # We want 100 points in a 10 by 10 array
    height, width = img.shape[:2]
    cell_side = (height - (border * 2)) / 9

    # Populate a 10 x 10 2D array with values that we have
    # Build up some objects with the points that are missing
    incomplete_rows = {}
    incomplete_cols = {}
    grid = []
    threshold = 0.3
    for i in range(10):
        grid.append([])
        for j in range(10):
            l_bound_y = ((i - threshold) * cell_side) + border
            u_bound_y = ((i + threshold) * cell_side) + border

            l_bound_x = ((j - threshold) * cell_side) + border
            u_bound_x = ((j + threshold) * cell_side) + border

            pos = list(filter(lambda point: l_bound_y <= point[1] < u_bound_y and l_bound_x <= point[0] < u_bound_x, points))

            # If we have more than one point in a segment, take an average of the positions
            if len(pos) > 1:
                pos = [np.mean([x[0] for x in pos]), np.mean([x[1] for x in pos])]
                grid[i].append(pos)
            elif len(pos) == 0:  # No points found
                # We know where the corners are, so can fill these in
                if i == 0 and j == 0:
                    grid[i].append([border, border])
                elif i == 9 and j == 0:
                    grid[i].append([border, height - border])
                elif i == 0 and j == 9:
                    grid[i].append([width - border, border])
                elif i == 9 and j == 9:
                    grid[i].append([width - border, height - border])
                else:  # Mark as incomplete
                    grid[i].append([])
                    incomplete_rows[i] = incomplete_rows.get(i, set())
                    incomplete_rows[i].add(j)
                    incomplete_cols[j] = incomplete_cols.get(j, set())
                    incomplete_cols[j].add(i)
            elif len(pos) == 1:
                grid[i].append([pos[0][0], pos[0][1]])

    def complete_cells():
        # Find the incomplete row or column that is nearest to completion
        min_idx = -1
        min_type = None
        min_count = 11

        # Check the rows
        for row_, cols in incomplete_rows.items():
            if len(cols) < min_count:
                min_count = len(cols)
                min_idx = row_
                min_type = 'row'
                if min_count == 1:
                    break

        # Check the columns
        for col, rows in incomplete_cols.items():
            if len(rows) < min_count:
                min_count = len(rows)
                min_idx = col
                min_type = 'col'
                if min_count == 1:
                    break

        # Completed the grid
        if min_count == 11:
            return True

        def least_squares(unit):
            """Calculates gradient and intercept of a line through the  available points using the least squares method."""
            non_zero = [pt for pt in unit if len(pt) > 0]
            mean_x = sum([pt[0] for pt in non_zero]) / len(non_zero)
            mean_y = sum([pt[1] for pt in non_zero]) / len(non_zero)
            numerator = sum([(pt[0] - mean_x) * (pt[1] - mean_y) for pt in non_zero])
            denominator = sum([(pt[0] - mean_x) ** 2 for pt in non_zero])

            if denominator == 0:
                m = 0
                c = -mean_x
            else:
                m = numerator / denominator
                c = mean_y - (m * mean_x)
            return m, c

        def check_cell(x, y):
            return 0 <= x <= width and 0 <= y <= height

        def fill_from_row(min_i):
            m, c = least_squares(grid[min_i])  # Get the line of best fit for the row
            for missing in incomplete_rows[min_i]:
                x = (missing * cell_side) + border  # Estimate x
                if m == 0 and c == 0:
                    y = (min_i * cell_side) + border  # Make a total guess based on the dimensions of the image
                else:
                    y = (m * x) + c  # Calculate using the line of best fit

                if len(grid[min_i][missing]) < 2 and check_cell(x, y):  # Don't override if already complete
                    grid[min_i][missing] = [x, y]

            # If the row is complete, remove it from our incomplete dictionary
            if all([len(c) == 2 for c in grid[min_i]]):
                del incomplete_rows[min_i]

        def fill_from_col(min_i):  # Fill in gaps for a column
            m, c = least_squares([grid[c][min_i] for c in range(10)])  # Get the line of best fit for the column
            for missing in incomplete_cols[min_i]:
                y = (missing * cell_side) + border  # Estimate y
                if m == 0 and c == 0:
                    x = (min_i * cell_side) + border  # Make a total guess based on the dimensions of the image
                else:
                    x = -c if m == 0 else (y - c) / m   # Calculate from the line of best fit

                if len(grid[missing][min_i]) < 2 and check_cell(x, y):  # Don't override if already complete
                    grid[missing][min_i] = [x, y]

            # If the column is complete, remove it from our incomplete dictionary
            if all([len(grid[c][min_i]) == 2 for c in range(10)]):
                del incomplete_cols[min_i]

        # Complete the row or column that is nearest to completion
        if min_type == 'row':
            fill_from_row(min_idx)
        elif min_type == 'col':
            fill_from_col(min_idx)

        complete_cells()  # Recurse until complete
    complete_cells()  # Complete the grid

    # Flatten grid into a simple list
    flattened = []
    for row in grid:
        for cell in row:
            if len(cell) == 2:
                flattened.append(cell)
    return flattened, grid


def extract_cell_raw(img, rect, size, include_gray_channel=False):
    """Extracts a cell from the image based on the rectangle and scales it to size."""
    cell = cv.cut_from_rect(img, rect)
    cell = cv2.resize(cell, (size, size))
    if include_gray_channel:
        cell = cell.reshape((size, size, 1))
    return cell


class Sudoku:
    def __init__(
            self, img_path=None, model_path=None, scale=1000, digit_size=28,
            include_gray_channel=False, skip_recog=False, skip_validate=False,
            frame=None,
    ):
        """
        Initialises a Sudoku Grid object from a photograph of a Sudoku board.
        Args:
            img_path (str): Path to the Sudoku photo.
            model_path (str): Path to the saved Tensorflow model used to recognise digits from images. Defaults to
                the path to `data/best-model/model.ckpt`.
            scale (int): Maximum allowed scale of any side of the image in pixels. Any images larger than this will be
                scaled down accordingly.
            digit_size (int): Size in pixels of the square that will be used for each digit.
            include_gray_channel (bool): Includes the grayscale channel when extracting digits, so the shape is:
                (digit_size, digit_size, 1) instead of (digit_size, digit_size). Useful when creating augmentations with
                the `imgaug` package.
            skip_recog (bool): Skips digit recognition.
            skip_validate (bool): Skips the validation algorithm if the board isn't solved immediately.
            frame(numpy.ndarray): Specify instead of img_path if you have a numpy array representation of an image.
        """
        self.img_path = img_path
        self.scale = scale
        self.digit_size = digit_size
        self.include_gray_channel = include_gray_channel
        self.border = 15

        if model_path is None:  # Default to the best model
            model_path = os.path.join(os.path.dirname(__file__), 'data', 'best-model', 'model.ckpt')

        if img_path is None and frame is None:
            raise ValueError("One of img_path or frame must be specified.")

        if img_path is not None and frame is not None:
            raise ValueError("Only one of img_path and frame must be specified")

        if img_path is not None and frame is None:
            if not os.path.exists(self.img_path):
                raise FileNotFoundError(self.img_path)

        # Read in the Sudoku photo
        if frame is not None:
            self.original = frame
        else:
            self.original = frame or cv2.imread(self.img_path, cv2.IMREAD_COLOR)
            
        # cv2.imshow('Original photo', self.original)
        self.original = cv.limit_scale(self.original, scale, scale)  # Scale down the image to speed up processing
        self.grayscale = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2GRAY)  # Grayscale

        # Crop and warp the image so it is just the Sudoku grid
        self.cropped, self.crop_rect, self.crop_matrix = crop_grid(self.grayscale)
        self.cropped_color = None  # Used later when showing digits on the board

        # Infer the grid points using Hough lines, linear algebra and estimation
        self.cropped, self.grid_points, self.grid_squares = infer_grid(self.cropped, border=self.border)

        # Use the neural network model to identify the digits
        self.digits = self.get_digits(include_blanks=True)

        self.board_int = [0] * 81

        if not skip_recog:
            digit_recogniser = DigitRecogniser(model_path)

            self.board_int = digit_recogniser.predict_digit(self.digits)


    def show_original(self):
        cv.show_image(self.original)

    def show_cropped(self):
        cv.show_image(self.cropped)


    def draw_numbers(self, numbers, colour=(50, 50, 255), thickness=3, crop=False, show_known=False, show=True, save=None):
        """
        Draws numbers onto the cropped or original Sudoku image and shows it on the screen.
        Args:
            numbers (iterable): Array of size 81 with the numbers to display on the grid.
            colour (tuple): BGR (blue, green, red) values between 0 and 255.
            thickness (int): Thickness of the font in pixels.
            crop (bool): If True, will display the numbers on the cropped and warped image instead of the original.
            show_known (bool): If True, will display the predictions for the given numbers on the board instead of the
                empty cells.
            show (bool): If True, will show the image on the screen.
            save (str): If specified, will save the image to that location.
        Returns:
            np.array: Image with the missing numbers drawn on.
        """
        if self.cropped_color is None:
            self.cropped_color, rect, matrix = cv.crop_and_warp(self.original, self.crop_rect, square=True)
        img = self.cropped_color.copy()

        scale = int((self.grid_squares[0][1][0] - self.grid_squares[0][0][0]) * 0.075)  # Dynamic scale for font
        for i, square in enumerate(self.grid_squares):
            condition = self.board_int[i] == 0  # Don't draw numbers given on the board

            if show_known:  # Unless we want to see them
                condition = not condition

            if condition:
                fh, fw = cv2.getTextSize(str(numbers[i]), cv2.FONT_HERSHEY_PLAIN, scale, thickness)[0]  # Get font height and width
                h_pad = int((square[1][0] - square[0][0] - fw) / 2)  # Padding to centre the number
                v_pad = int((square[1][1] - square[0][1] - fw) / 2)
                h_pad -= self.border  # No border on the original, so this compensates
                v_pad += self.border
                img = cv2.putText(img, str(numbers[i]), (int(square[0][0]) + h_pad, int(square[1][1]) - v_pad),
                                  cv2.FONT_HERSHEY_PLAIN, fontScale=scale, color=colour, thickness=thickness)

        # Display the cropped image and return
        if crop:
            if show:
                cv.show_image(img)
            if save is not None:
                cv2.imwrite(save, img)
            return img

        # Perform an inverse of the crop and warp transformation to put the image back onto the original
        height, width = self.original.shape[:2]
        img = cv2.warpPerspective(img, self.crop_matrix, (width, height), flags=cv2.WARP_INVERSE_MAP, dst=self.original,
                                  borderMode=cv2.BORDER_TRANSPARENT)
        if show:
            cv.show_image(img)

        if save is not None:
            cv2.imwrite(save, img)
        return img


    def show_completed(self, colour=(50, 50, 255), thickness=3, show=True, save=None):
        """Shows the cropped image with the solution overlaid on the missing boxes."""
        if self.solution is None:  # When we know we haven't got a solution, don't continue
            return None
        else:
            answers = [v for k, v in solver.parse_sudoku_puzzle(self.solution).items()]  # Solution as a list
            return self.draw_numbers(answers, colour, thickness, show=show, save=save)


    def get_digits(self, save=None, show=False, include_blanks=False, raw=False):
        """
        Saves only the extracted digits from a Sudoku grid to the digits subfolder.
        Args:
            save (str): If specified, will save the extracted squares as JPG to the target directory with images named
                according to the 0 based grid position. If the directory already exists, the results will be overwritten.
            show (bool): If True, will display each square as they are extracted.
            include_blanks (bool): If True, will include squares where a digit could not be found as a completely black square.
            raw (bool): If True, will extract the digits without any pre-processing.
        Returns:
            np.array: Array of all `np.array` images for each digit in the Sudoku grid.
        """
        blank = cv.create_blank_image(self.digit_size, self.digit_size, include_gray_channel=self.include_gray_channel)

        digits = []

        for i, rect in enumerate(self.grid_squares):
            digit = extract_cell_raw(self.cropped, rect, self.digit_size, include_gray_channel=self.include_gray_channel)

            if digit is not None:
                digits.append(digit)

                if save is not None:
                    cv2.imwrite(os.path.join(save, '%s.jpg' % str(i)), digit)

                if show:
                    cv.show_image(digit)
            elif include_blanks:
                digits.append(blank)

        return np.array(digits)

    @property
    def board(self):
        return solver.display_sudoku_grid(list(self.board_int))

    @property
    def board_dict(self):
        return solver.parse_sudoku_puzzle(self.board)

    @property
    def board_str(self):
        return [str(x) if x != 0 else '.' for x in self.board_int]

    @property
    def solution(self):
        answer = solver.solve(self.board)
        if solver.validate_sudoku(answer):
            return solver.display_sudoku_grid(answer)
        else:
            return None