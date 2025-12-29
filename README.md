## 🌲 Adaptive AVL Tree & Finger Search Optimization
Turning O(logn) into O(1) for sorted data streams.
A Python implementation of a self balancing AVL Tree augmented with Finger Search mechanics.
This project explores "Adaptive Sorting" by proving both theoretically and experimentally that tree operations can achieve amortized O(1) complexity when
the input data exhibits pre existing order (low entropy).

## 🚀 Key Features
Finger Search & Insert: Unlike standard BSTs that always traverse from the root, this implementation maintains a dynamic pointer ("finger") to the maximum 
element. This allows for O(1) access time for sequential data.
Adaptive Complexity: The algorithm "adapts" to the disorder of the input.
Random Input: Standard O(logn).
Sorted/Nearly Sorted Input: Amortized O(1).
Advanced Topology Operations: Supports split() and join() operations in logarithmic time O(logn), enabling efficient merging and partitioning of data
structures. Robust Architecture: Built with Virtual Sentinel Nodes to eliminate null pointer checks and simplify rotation logic (LL, RR, LR, RL).
Pure Python: Implemented in Python without external data structure libraries.

## 📊 Theoretical Analysis
The core innovation of this project is the correlation between the Number of Inversions (I) in the input sequence and the total runtime.
We proved that for a sequence of n insertions, the total search cost is bounded by:
Total Cost = O(n⋅log((n/I)+2)).
What does this mean?
Sorted Array (I=0): The cost becomes O(n⋅log2) = O(n). (Linear time total, O(1) per op).
Reverse Sorted (I≈(n^2)/2): The cost becomes O(nlogn). (Standard worst case).
Local Chaos (I≈n): For data with only local swaps, runtime remains linear O(n).
Full mathematical proof and amortized analysis are available in 'AVLTree_Research'.

The experiments confirmed that Finger Search successfully exploits locality of reference, making it superior for real-world time series data or logs.

## 🛠️ Usagepython
from AVLTree import AVLTree

#Initialize the tree
tree = AVLTree()

#Standard Insert (starts from root)
tree.insert(key=15, value="A")

#Finger Insert (starts from max node - efficient for sorted data)
tree.finger_insert(key=16, value="B") tree.finger_insert(key=17, value="C") # O(1) operation!

#Search
node, cost = tree.search(15)

#Split the tree at key 16
t1, t2 = tree.split(node)

## Co-authored with Yael Aviram.

## Focus: *Algorithm Design, Amortized Analysis, Python Optimization.*

---
