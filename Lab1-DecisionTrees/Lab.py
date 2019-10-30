import monkdata as m
import dtree as d
import matplotlib.pyplot as plt
import numpy as np
import random

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


"""
print(d.entropy(m.monk1))
print(d.entropy(m.monk2))
print(d.entropy(m.monk3))


print("#################")
print(averageGain(m.monk1, m.attributes[0]))
print(averageGain(m.monk1, m.attributes[1]))
print(averageGain(m.monk1, m.attributes[2]))
print(averageGain(m.monk1, m.attributes[3]))
print(averageGain(m.monk1, m.attributes[4]))
print(averageGain(m.monk1, m.attributes[5]))

print("#################")
print(averageGain(m.monk2, m.attributes[0]))
print(averageGain(m.monk2, m.attributes[1]))
print(averageGain(m.monk2, m.attributes[2]))
print(averageGain(m.monk2, m.attributes[3]))
print(averageGain(m.monk2, m.attributes[4]))
print(averageGain(m.monk2, m.attributes[5]))

print("#################")
print(averageGain(m.monk3, m.attributes[0]))
print(averageGain(m.monk3, m.attributes[1]))
print(averageGain(m.monk3, m.attributes[2]))
print(averageGain(m.monk3, m.attributes[3]))
prin t(averageGain(m.monk3, m.attributes[4]))
print(averageGain(m.monk3, m.attributes[5]))


FirstNode_1 = d.select(m.monk1, m.attributes[4], 1)


print("node = 1")
print(d.averageGain(FirstNode_1, m.attributes[0]))
print(d.averageGain(FirstNode_1, m.attributes[1]))
print(averageGain(FirstNode_1, m.attributes[2]))
print(averageGain(FirstNode_1, m.attributes[3]))
print(averageGain(FirstNode_1, m.attributes[4]))
print(averageGain(FirstNode_1, m.attributes[5]))

print("node = 2")
FirstNode_2 = select(m.monk1, m.attributes[4], 2)
print(averageGain(FirstNode_2, m.attributes[0]))
print(averageGain(FirstNode_2, m.attributes[1]))
print(averageGain(FirstNode_2, m.attributes[2]))
print(averageGain(FirstNode_2, m.attributes[3]))
print(averageGain(FirstNode_2, m.attributes[4]))
print(averageGain(FirstNode_2, m.attributes[5]))

print("node = 3")
FirstNode_3 = select(m.monk1, m.attributes[4], 3)
print(averageGain(FirstNode_3, m.attributes[0]))
print(averageGain(FirstNode_3, m.attributes[1]))
print(averageGain(FirstNode_3, m.attributes[2]))
print(averageGain(FirstNode_3, m.attributes[3]))
print(averageGain(FirstNode_3, m.attributes[4]))
print(averageGain(FirstNode_3, m.attributes[5]))

print("node = 4")
FirstNode_4 = select(m.monk1, m.attributes[4], 4)

print(averageGain(FirstNode_4, m.attributes[0]))
print(averageGain(FirstNode_4, m.attributes[1]))
print(averageGain(FirstNode_4, m.attributes[2]))
print(averageGain(FirstNode_4, m.attributes[3]))
print(averageGain(FirstNode_4, m.attributes[4]))
print(averageGain(FirstNode_4, m.attributes[5]))



noFirstNode = exclude(m.monk1, m.attributes[4], 1)

print(averageGain(noFirstNode, m.attributes[0]))
print(averageGain(noFirstNode, m.attributes[1]))
print(averageGain(noFirstNode, m.attributes[2]))
print(averageGain(noFirstNode, m.attributes[3]))
print(averageGain(noFirstNode, m.attributes[4]))
print(averageGain(noFirstNode, m.attributes[5]))

print(mostCommon(FirstNode_1))
print(mostCommon(FirstNode_2))
print(mostCommon(FirstNode_3))
print(mostCommon(FirstNode_4))

qt.drawTree(buildTree(m.monk1, m.attributes, 2))


t=d.buildTree(m.monk1, m.attributes)
print(d.check(t, m.monk1))
print(d.check(t, m.monk1test))

t=d.buildTree(m.monk2, m.attributes)
print(d.check(t, m.monk2))
print(d.check(t, m.monk2test))

t=d.buildTree(m.monk3, m.attributes)
print(d.check(t, m.monk3))
print(d.check(t, m.monk3test))


"""



def getPrunedTree(initialTree, validationSet):
    continuePruning = True
    currentTree = initialTree
    while(continuePruning):
        currentTreeScore = d.check(currentTree, validationSet)
        prunedTrees = d.allPruned(currentTree)
        bestPrunedTreeScore = 0
        for idx_tree in range(len(prunedTrees)):
            tree = prunedTrees[idx_tree]
            score = d.check(tree, validationSet)
            if score > bestPrunedTreeScore:
                bestPrunedTreeScore = score
                bestPrunedTree_idx = idx_tree
        if bestPrunedTreeScore < currentTreeScore:
            continuePruning = False
        else:
            currentTree = prunedTrees[bestPrunedTree_idx]


    return currentTree



def test_error_per_partition(monk, monktest):
    fraction_values = [0.3, 0.4, 0.5, 0.6, 0.7,0.8]
    test_error_mean = []
    test_error_std = []
    for partition_number in fraction_values:
        print(partition_number)
        testErrors_list =[]
        n_iters = 600
        for iter in range(n_iters):
            monktrain, monkval = partition(monk, partition_number)
            tree = d.buildTree(monktrain, m.attributes)
            prunedTree = getPrunedTree(tree, monkval)
            testError = 1 - d.check(prunedTree, monktest)
            testErrors_list.append(testError)
        print("all iters calculated")
        testErrors_np = np.array(testErrors_list)
        test_error_mean.append(testErrors_np.mean())
        test_error_std.append(testErrors_np.std())


    plt.scatter(fraction_values, test_error_mean, c=test_error_std)
    cbar = plt.colorbar()
    cbar.set_label('Standard deviation', rotation=270, labelpad=30)
    plt.xlabel("fraction parameter")
    plt.ylabel("Average classification error (test set)")
    plt.show()


    # Hay que hacer la media y la varianza
    # test_error_per_partition
    # PLOTEAR
    # plt.plot(fraction values, test_error_per_partition)
    #



test_error_per_partition(m.monk3, m.monk3test)
