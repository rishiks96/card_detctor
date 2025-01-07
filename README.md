# Playing Card Detector

A real-time playing card detection system that identifies playing cards using your computer's webcam.

## Description

This application uses computer vision to detect and identify playing cards in real-time. It captures video from your webcam, processes each frame to find cards, and matches them against reference images to identify the card's value and suit.

## Requirements

- Python 3.7 or higher
- OpenCV library
- NumPy library
- Queue (Python standard library)

## Setup

Make sure you have Python installed on your computer. Install the required libraries using pip. Place your reference card images in a folder named 'reference_cards'.

## How to Use

1. Connect a webcam to your computer
2. Run the main program
3. Hold a playing card in front of the camera
4. The program will show the video feed and display the detected card's value and suit

## Features

- Real-time card detection and recognition
- Works with standard playing cards
- Shows live video feed with detection results
- Uses efficient image processing for quick recognition

## Notes

- Good lighting will improve detection accuracy
- Cards should be clearly visible and not bent
- Keep the card steady for better recognition
- The program works best with a plain, dark background

