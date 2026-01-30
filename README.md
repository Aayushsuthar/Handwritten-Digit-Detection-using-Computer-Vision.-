# Handwritten Digit Recognition

#### Building a Neural Network from Scratch (no TensorFlow/PyTorch, just NumPy & Math)

## Author

**Aayush Suthar** (he/him)

👋 Hi, I'm Aayush Suthar, a Computer Science and Engineering undergraduate specializing in Artificial Intelligence and Machine Learning at Manipal University.

- 📍 Jaipur, India
- 📧 aayushsuthar5115@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/aayushsuthar)
- 🐦 [Twitter](https://twitter.com/aayushsuthar711)
- 💻 [GitHub](https://github.com/Aayushsuthar)
- 🧩 [LeetCode](https://leetcode.com/u/Aayush_Suthar/)

## General Overview

During this semester, I studied the **Artificial Intelligence** curriculum and learned about **Artificial Neural Networks**. While I had some experience with Python and machine learning basics, I was still new to computer vision. Building this project from scratch was a perfect introduction to **Computer Vision** using artificial neural networks (ANN).

## What is Handwritten Digit Recognition?

Handwritten digit recognition is the ability of computers to recognize human handwritten digits. It is a challenging task for machines because handwritten digits are not perfect and can be written in many different styles. This project uses images of digits and recognizes which digit is present in each image.

---

## 1. Problem Statement & Objective

- My goal is to correctly identify digits from a dataset of tens of thousands of handwritten images.
- The dataset is provided by **MNIST** ("Modified National Institute of Standards and Technology"), one of the most popular datasets in computer vision.
  - This classic dataset of handwritten images has served as the basis for benchmarking classification algorithms.
  - As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

![MNIST Dataset Sample 1](https://user-images.githubusercontent.com/54215462/174850145-5140e711-3da7-49d3-859c-9ca42402530c.png)

![MNIST Dataset Sample 2](https://user-images.githubusercontent.com/54215462/174850313-f22b9856-500c-4f24-8dc6-e09e261b99d1.png)

## 2. Math Explanation

I implemented a two-layer artificial neural network and trained it on the MNIST digit recognizer dataset. It's implemented from scratch in Python using only the NumPy library—to deal with lists/matrices—and basic math. This approach helped me understand the underlying mathematics of neural networks better.

![Neural Network Architecture](https://user-images.githubusercontent.com/54215462/174853072-9ea6f6a5-e4f3-49c8-9420-815d9a64894a.png)

### Neural Network Architecture

- **Input layer**: 784 units/neurons corresponding to the 784 pixels in each 28×28 input image
- **Hidden layer**: 10 neurons with ReLU activation
- **Output layer**: 10 output units (representing possible classifications from 0 to 9) with Softmax activation

### Mathematical Foundations

![Forward Propagation](https://user-images.githubusercontent.com/54215462/174853244-09a4c88c-9600-4b19-a9ae-cca61f69e9a3.png)

![Activation Functions](https://user-images.githubusercontent.com/54215462/174853299-b19d3f08-f93b-4337-826e-ba02dd0efa30.png)

![Cost Function](https://user-images.githubusercontent.com/54215462/174853334-bd019130-912d-45ca-a266-f2247ad5696e.png)

![Backpropagation](https://user-images.githubusercontent.com/54215462/174853373-a177016d-b892-442c-9d5b-10cb1f2883f7.png)

![Gradient Descent](https://user-images.githubusercontent.com/54215462/174854176-0fa94878-c18f-4b4c-b2cb-cde88e4ce674.png)

![Parameter Update](https://user-images.githubusercontent.com/54215462/174853454-7da0060b-f6cd-417b-8641-1354cbe509ba.png)

![Matrix Dimensions](https://user-images.githubusercontent.com/54215462/174853472-dfe67fce-76f1-4dc3-a033-ac3c5af3bf3b.png)

![Derivatives](https://user-images.githubusercontent.com/54215462/174853818-bea905d9-2416-43a9-b817-9436f000b180.png)

![Chain Rule](https://user-images.githubusercontent.com/54215462/174853886-f84e9745-05ff-4290-a82e-1f9d65e7faea.png)

## 3. Coding It Up

Here is a glimpse of the training set used to train our neural network:

![Training Set](https://user-images.githubusercontent.com/54215462/174854320-072646ef-41f3-4acb-8d41-2ca0fab80dc7.png)

You can check out the file `code.py` for the complete implementation 😁

## 4. Results

### Training Progress

![Training Iteration 1](https://user-images.githubusercontent.com/54215462/174854726-60f6e1d0-9153-46a9-8162-565b99dc98d0.png)

![Training Iteration 2](https://user-images.githubusercontent.com/54215462/174854765-e93b7016-7243-4110-8a27-401fab737fc0.png)

![Training Iteration 3](https://user-images.githubusercontent.com/54215462/174854803-0d02b087-676c-4639-9a95-835570d8205c.png)

![Training Iteration 4](https://user-images.githubusercontent.com/54215462/174854867-c88b0868-8709-4dfb-a521-0abbf215a040.png)

...

![Final Training Results](https://user-images.githubusercontent.com/54215462/174854911-9061c8c7-60f1-4cec-abfc-dd1817e08739.png)

![Testing Accuracy](https://user-images.githubusercontent.com/54215462/174854942-bd72217f-0549-4429-b59a-d0c5ce4f72b0.png)

### Sample Predictions

Let's look at a couple of examples:

![Sample Predictions Overview](https://user-images.githubusercontent.com/54215462/174855235-b416c89a-7964-4488-86aa-5e3d7f7f1b78.png)

![Prediction Example 1](https://user-images.githubusercontent.com/54215462/174855280-3117ad7f-5d8b-44af-aa79-1b34db56411b.png)
![Prediction Example 2](https://user-images.githubusercontent.com/54215462/174855128-2a795f1d-ba75-4949-8dc1-8b4d985dde25.png)
![Prediction Example 3](https://user-images.githubusercontent.com/54215462/174855332-3f5826cc-538f-4e21-ae5d-073012102630.png)
![Prediction Example 4](https://user-images.githubusercontent.com/54215462/174855364-d02c5c50-a449-4caa-a18d-0acfc830aabc.png)

![Final Results](https://user-images.githubusercontent.com/54215462/174855433-69381952-a3d4-498e-b331-cf2545d4c016.png)

## Key Features

- **From Scratch Implementation**: No high-level frameworks like TensorFlow or PyTorch
- **Pure NumPy**: All matrix operations and calculations done with NumPy
- **Educational Focus**: Clear code with detailed comments explaining each step
- **Gradient Descent Optimization**: Manual implementation of backpropagation and parameter updates
- **Visualization**: Built-in functions to visualize predictions and compare with actual labels

## How to Run

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have the MNIST dataset in a `data/` folder with `train.csv`

3. Run the training script:
   ```bash
   python code.py
   ```

## Project Structure

- `code.py`: Main implementation file containing the neural network
- `README.md`: This file
- `requirements.txt`: Python dependencies
- `data/train.csv`: MNIST training dataset (not included, must be downloaded separately)

## Learning Outcomes

Through this project, I gained a deep understanding of:

- Forward and backward propagation
- Activation functions (ReLU and Softmax)
- Gradient descent optimization
- Matrix operations for efficient computation
- The mathematics behind neural networks

---

And that's it! If you found this repository interesting, kindly give it a star ⭐😉

## Acknowledgments

- MNIST Dataset providers
- The broader machine learning community for educational resources
