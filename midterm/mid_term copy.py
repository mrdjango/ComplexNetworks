{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "import networkx as nx\n",
    "import numpy as np\n",
    "from node2vec import Node2Vec\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.pipeline import make_pipeline\n",
    "from sklearn.model_selection import cross_val_score, KFold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "graph_list = []\n",
    "for i in range(30):\n",
    "    graph_list.append((nx.gnp_random_graph(10,0.9, seed=None,directed = False), '1'))\n",
    "    graph_list.append((nx.gnp_random_graph(10,0.1, seed=None,directed = False), '0'))\n",
    "\n",
    "graph_list = np.array(graph_list, dtype=[('graph', object), ('label', object)])\n",
    "G = graph_list.reshape(60, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1119.50it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 288.81it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 339.96it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 277.34it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 407.48it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 23096.39it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 1642.41it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 1559.80it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 1185.66it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 1070.34it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1619.30it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 299.83it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 206.68it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 273.61it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 173.99it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 15984.39it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 580.65it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 551.71it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 325.19it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 261.09it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1779.81it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 346.58it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 234.98it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 209.52it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 294.09it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 19925.43it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 692.49it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 586.94it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 501.22it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 429.32it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 3511.05it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 310.23it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 367.15it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 294.75it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 262.20it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 10369.11it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 665.49it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 521.97it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 394.60it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 385.94it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 2568.15it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 368.95it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 324.37it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 263.01it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 342.88it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 12087.33it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 517.00it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 451.62it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 506.24it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 339.65it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1431.94it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 375.92it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 262.54it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 267.35it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 447.24it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 13586.99it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 617.17it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 519.13it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 370.07it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 394.10it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1638.78it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 397.14it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 366.12it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 340.16it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 374.16it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 18228.18it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 1158.40it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 1100.07it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 903.62it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 792.34it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 2238.75it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 329.46it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 345.54it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 325.10it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 504.65it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 14691.08it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 656.78it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 552.01it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 508.09it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 492.53it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1512.39it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 422.11it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 418.72it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 288.43it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 431.27it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 17147.60it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 747.67it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 545.46it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 484.56it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 521.49it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1234.05it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 341.02it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 286.18it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 388.24it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 482.11it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 18032.26it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 807.03it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 803.74it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 534.35it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 487.12it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 2975.11it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 416.11it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 304.30it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 289.10it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 415.92it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 14538.32it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 621.10it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 512.55it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 620.26it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 434.98it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1480.62it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 412.76it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 258.85it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 244.18it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 388.01it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 21076.90it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 944.12it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 785.23it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 673.27it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 608.33it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1600.21it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 298.99it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 432.21it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 303.77it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 339.54it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 8971.77it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 645.74it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 424.89it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 424.75it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 411.10it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1659.80it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 403.07it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 406.68it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 308.71it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 609.70it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 27575.96it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 771.69it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 652.97it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 471.40it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 331.80it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 2320.89it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 385.84it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 367.68it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 804.98it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 468.05it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 410.66it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 2137.33it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 335.24it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 362.85it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 325.42it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 481.34it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 11755.34it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 612.45it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 520.24it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 426.00it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 372.17it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1696.86it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 367.82it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 307.96it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 314.83it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 403.22it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 9838.86it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 692.62it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 549.93it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 372.13it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 366.92it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 1835.66it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 403.13it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 262.33it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 268.47it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 431.71it/s]\n",
      "Computing transition probabilities: 100%|██████████| 10/10 [00:00<00:00, 9576.04it/s]\n",
      "Generating walks (CPU: 3): 100%|██████████| 12/12 [00:00<00:00, 475.51it/s]\n",
      "Generating walks (CPU: 4): 100%|██████████| 12/12 [00:00<00:00, 581.52it/s]\n",
      "Generating walks (CPU: 1): 100%|██████████| 13/13 [00:00<00:00, 436.74it/s]\n",
      "Generating walks (CPU: 2): 100%|██████████| 13/13 [00:00<00:00, 430.43it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(60, 10, 1, 1)\n"
     ]
    }
   ],
   "source": [
    "# Extract graphs from the list\n",
    "graphs = [graph[0] for graph in graph_list]\n",
    "\n",
    "# Generate node embeddings for each graph using node2vec\n",
    "embeddings_list = []\n",
    "for graph in graphs:\n",
    "    node2vec = Node2Vec(graph, dimensions=1, walk_length=30, num_walks=50, workers=4)\n",
    "    model = node2vec.fit(window=10, min_count=1, batch_words=4)\n",
    "    embeddings = [[model.wv.get_vector(str(node))] for node in graph.nodes]\n",
    "    embeddings_list.append(embeddings)\n",
    "\n",
    "# Convert the embeddings list to a NumPy array\n",
    "node_embeddings = np.array(embeddings_list)\n",
    "print(node_embeddings.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60, 10)"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# node_list = [i[:,0,0] for i in node_embeddings]\n",
    "# l = np.array(node_list)\n",
    "# l.shape\n",
    "l = node_embeddings[:,:,0,0]\n",
    "l.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame(l)\n",
    "df[10] = np.array(graph_list)['label']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "features = [0,1,2,3,4,5,6,7,8,9]\n",
    "X = df.loc[:, features]\n",
    "Y = df.loc[:,10]\n",
    "x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42,  test_size=0.2,  shuffle=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9166666666666666\n"
     ]
    }
   ],
   "source": [
    "model = DecisionTreeClassifier(random_state=22)\n",
    "model.fit(x_train,y_train)\n",
    "\n",
    "cross_val = KFold(n_splits=30, random_state=42,shuffle=True)\n",
    "scores = cross_val_score(model, X, Y, cv= cross_val, n_jobs=-1)\n",
    "print(np.abs(np.mean(scores)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9833333333333333\n"
     ]
    }
   ],
   "source": [
    "model = make_pipeline(StandardScaler(), SVC(gamma='auto'))\n",
    "model.fit(x_train,y_train)\n",
    "\n",
    "cross_val = KFold(n_splits=30, random_state=42,shuffle=True)\n",
    "scores = cross_val_score(model, X, Y, cv= cross_val, n_jobs=-1)\n",
    "print(np.abs(np.mean(scores)))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
