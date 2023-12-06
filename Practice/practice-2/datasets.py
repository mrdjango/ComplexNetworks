"""
This directory contains the TSV and related files of the ucidata-zachary network:
    This is the well-known and much-used Zachary karate club network.
    The data was collected from the members of a university karate club by Wayne Zachary in 1977.
    Each node represents a member of the club, and each edge represents a tie between two members of the club.
    The network is undirected.
    An often discussed problem using this dataset is to find the two groups of people into which the karate club split after an argument between two teachers. 

zachary dataset structure: 
    meta.ucidata-zachary -- Metadata about the network 
    out.ucidata-zachary -- The adjacency matrix of the network in whitespace-separated values format,
        with one edge per line
      The meaning of the columns in out.ucidata-zachary are: 
        First column: ID of from node 
        Second column: ID of to node
        Third column (if present): weight or multiplicity of edge
        Fourth column (if present):  timestamp of edges Unix time
"""

from pathlib import Path


def zachary_dataset():
    file_path = f"{Path(__file__).parent.as_posix()}/ucidata-zachary.dat"

    # خواندن داده‌ها از فایل و تبدیل به آرایه
    data = []
    with open(file_path, "r+") as file:
        for line in file:
            # جداکردن دو عدد از هر خط و تبدیل به اعداد صحیح
            numbers = [int(x) for x in line.strip().split() if x.isdigit()]
            if len(numbers) == 2:
                data.append(numbers)

    return data


if __name__ == "__main__":
    print(zachary_dataset())
