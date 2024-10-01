# Shape Analysis and Optimization for 3D Printing

## Project Overview

This project focuses on addressing the balance problem in 3D printed models. Inspired by the paper "Make It Stand: Balancing Shapes for 3D Fabrication," we explore methods for internal carving using voxelization to shift the center of gravity, ensuring balanced models without the need for additional support structures.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Problem Modelling](#problem-modelling)
- [Internal Carving and Voxelization](#internal-carving-and-voxelization)
- [Implementation](#implementation)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

## Introduction

3D printing is widely used across various industries, but maintaining balance in printed objects can be a challenge. Many models require additional support to stay upright. This project seeks to address this issue by manipulating the internal structure of models to shift their center of gravity, ensuring balance without external supports.

## Objectives

- Reproduce the method proposed in "Make It Stand: Balancing Shapes for 3D Fabrication."
- Modify the internal structure of 3D models to shift the center of gravity into the support polygon.
- Implement voxelization techniques to simplify internal sculpting and model processing.
- Experiment with high-density material filling as an alternative solution for achieving balance.

## Problem Modelling

To keep a model balanced, the projection of its center of gravity must lie within the support polygon. By altering the internal structure of the model through voxelized sculpting, we can adjust the center of gravity to fall within this polygon, ensuring balance. This project uses convex hull-based support polygons to calculate and manage the center of gravity.

## Internal Carving and Voxelization

Voxelization is used to simplify internal carving by dividing the model into smaller units. This allows for efficient manipulation of the internal structure while maintaining the overall shape of the model. Our approach focuses on voxelized internal carving, making the process faster and more manageable for large models.

## Implementation

- **Programming Language**: Python
- **Input**: The program accepts `.obj` files and a user-defined support plane (specified by three points).
- **Process**:
  1. Read the input `.obj` file and store the model data.
  2. Accept user input to define a support plane.
  3. Rotate and translate the model to align the support plane with the xOy plane.
  4. Voxelize the model with high accuracy.
  5. Cut the model based on the support plane and remove voxels below it.
  6. Calculate the center of gravity and verify its alignment within the support polygon.
  7. Perform voxelized internal sculpting to shift the center of gravity if necessary.
  8. Output the balanced model in `.stl` format.

- **Challenges**: During development, we faced issues with the precision of cuts on the triangle mesh, leading to inaccuracies in the resulting models. To overcome this, we changed the approach to voxelize the entire model before performing cuts.

## Results

We tested our method on several models, including spheres and gargoyles. The program successfully generated internally sculpted models with adjusted centers of gravity, ensuring they remain balanced after printing. However, in some cases, additional sculpting was required to achieve stability.

## Future Improvements

- **Multi-material 3D Printing**: In future versions, we aim to incorporate high-density materials into the sculpting process. By filling unsculpted sections with dense materials, we can further shift the center of gravity without altering the external appearance of the model.
- **Deformation Techniques**: While not explored in this project, future work could investigate shape deformation as a way to balance more complex models. This would involve further research into non-intrusive deformation methods.

## Conclusion

This project successfully implements voxelized internal carving to balance 3D models for printing. While challenges were encountered with mesh precision, voxelization proved to be a viable solution. Moving forward, we plan to explore additional techniques such as multi-material printing and deformation to further enhance the balance of 3D printed models.
