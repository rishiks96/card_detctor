import cv2
import numpy as np
import os
import json
from datetime import datetime
import time
import threading
from queue import Queue


class CardDetectionSystem:
    def __init__(self, ref_dir='reference_images'):


        # Card detection parameters
        self.min_card_area = 20000
        self.max_card_area = 120000

        # Load reference images
        self.load_reference_images(ref_dir)

        # Storage setup
        self.detected_cards = []


    def load_reference_images(self, ref_dir):
        Load reference images with error handling
        self.rank_templates = {}
        self.suit_templates = {}

        try:
            # Load ranks
            rank_dir = os.path.join(ref_dir, 'ranks')
            if not os.path.exists(rank_dir):
                raise FileNotFoundError(f"Rank directory not found: {rank_dir}")

            for rank in ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']:
                path = os.path.join(rank_dir, f"{rank}.jpg")
                if os.path.exists(path):
                    self.rank_templates[rank] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # Load suits
            suit_dir = os.path.join(ref_dir, 'suits')
            if not os.path.exists(suit_dir):
                raise FileNotFoundError(f"Suit directory not found: {suit_dir}")

            for suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades']:
                path = os.path.join(suit_dir, f"{suit}.jpg")
                if os.path.exists(path):
                    self.suit_templates[suit] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        except Exception as e:
            print(f"Error loading reference images: {str(e)}")
            raise

    def match_template(self, region, templates, threshold=0.6):
        best_match = None
        best_score = threshold

        for name, template in templates.items():
            res = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
            if res.max() > best_score:
                best_match = name
                best_score = res.max()

        return best_match if best_score >= threshold else None

    def find_cards(thresh_image):
        Finds all card-sized contours in a thresholded camera image.
        Returns the number of cards, and a list of card contours sorted
        from largest to smallest.

         Find contours and sort their indices by contour size
        dummy, cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

         If there are no contours, do nothing
        if len(cnts) == 0:
            return [], []

         Otherwise, initialize empty sorted contour and hierarchy lists
        cnts_sort = []
        hier_sort = []
        cnt_is_card = np.zeros(len(cnts), dtype=int)

        # Fill empty lists with sorted contour and sorted hierarchy. Now,
        # the indices of the contour list still correspond with those of
        # the hierarchy list. The hierarchy array can be used to check if
        # the contours have parents or not.
        for i in index_sort:
            cnts_sort.append(cnts[i])
            hier_sort.append(hier[0][i])

        # Determine which of the contours are cards by applying the
        # following criteria: 1) Smaller area than the maximum card size,
        # 2), bigger area than the minimum card size, 3) have no parents,
        # and 4) have four corners

        for i in range(len(cnts_sort)):
            size = cv2.contourArea(cnts_sort[i])
            peri = cv2.arcLength(cnts_sort[i], True)
            approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)

            if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
                    and (hier_sort[i][3] == -1) and (len(approx) == 4)):
                cnt_is_card[i] = 1

        return cnts_sort, cnt_is_card

    def get_card_image(self, frame, contour):
        "Extract the card image from the frame based on the contour"
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        # Calculate the width and height of the card
        width = int(rect[1][0])
        height = int(rect[1][1])

        # Ensure the card is not too small or too large
        if width < 50 or height < 70 or width > 300 or height > 450:
            return None

        # Rotate the card to be upright
        angle = rect[2]
        rows, cols = frame.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(frame, M, (cols, rows))

        # Extract the card from the rotated frame
        x, y = box[0]
        # Check if the card is within the frame boundaries
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + width > cols:
            width = cols - x
        if y + height > rows:
            height = rows - y
        card_img = rotated[y:y + height, x:x + width]
        return card_img 

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_cards = []
        for contour in contours:
            if self.min_card_area < cv2.contourArea(contour) < self.max_card_area:
                card_info = self.process_card(frame, contour)
                if card_info:
                    detected_cards.append(card_info)
                    self.draw_card_info(frame, contour, card_info)

        return frame, detected_cards

    def process_card(self, frame, contour):
        """Process individual card with improved template matching"""
        # Get card image
        card_img = self.get_card_image(frame, contour)
        if card_img is None:
            return None

        # Extract and identify regions
        rank_region = card_img[5:85, 5:50], cv2.COLOR_BGR2GRAY
        suit_region = card_img[55:120, 5:50], cv2.COLOR_BGR2GRAY

        rank = self.match_template(rank_region, self.rank_templates, threshold=0.6)
        suit = self.match_template(suit_region, self.suit_templates, threshold=0.6)

        if rank and suit:
            return {'rank': rank, 'suit': suit, 'timestamp': datetime.now().isoformat()}
        return None

    def preprocess_card(contour, image):
        """Uses contour to find information about the query card. Isolates rank
        and suit images from the card."""

        # Initialize new Query_card object
        qCard = Query_card()

        qCard.contour = contour

        # Find perimeter of card and use it to approximate corner points
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        pts = np.float32(approx)
        qCard.corner_pts = pts

        # Find width and height of card's bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        qCard.width, qCard.height = w, h

        # Find center point of card by taking x and y average of the four corners.
        average = np.sum(pts, axis=0) / len(pts)
        cent_x = int(average[0][0])
        cent_y = int(average[0][1])
        qCard.center = [cent_x, cent_y]

        # Warp card into 200x300 flattened image using perspective transform
        qCard.warp = flattener(image, pts, w, h)

        # Grab corner of warped card image and do a 4x zoom
        Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
        Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)

        # Sample known white pixel intensity to determine good threshold level
        white_level = Qcorner_zoom[15, int((CORNER_WIDTH * 4) / 2)]
        thresh_level = white_level - CARD_THRESH
        if (thresh_level <= 0):
            thresh_level = 1
        retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

        # Split in to top and bottom half (top shows rank, bottom shows suit)
        Qrank = query_thresh[20:185, 0:128]
        Qsuit = query_thresh[186:336, 0:128]

        # Find rank contour and bounding rectangle, isolate and find largest contour
        dummy, Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)

        # Find bounding rectangle for largest contour, use it to resize query rank
        # image to match dimensions of the train rank image
        if len(Qrank_cnts) != 0:
            x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
            Qrank_roi = Qrank[y1:y1 + h1, x1:x1 + w1]
            Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
            qCard.rank_img = Qrank_sized

        # Find suit contour and bounding rectangle, isolate and find largest contour
        dummy, Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea, reverse=True)

        # Find bounding rectangle for largest contour, use it to resize query suit
        # image to match dimensions of the train suit image
        if len(Qsuit_cnts) != 0:
            x2, y2, w2, h2 = cv2.boundingRect(Qsuit_cnts[0])
            Qsuit_roi = Qsuit[y2:y2 + h2, x2:x2 + w2]
            Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
            qCard.suit_img = Qsuit_sized

        return qCard

    def draw_card_info(self, frame, contour, card_info):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{card_info['rank']} of {card_info['suit']}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



